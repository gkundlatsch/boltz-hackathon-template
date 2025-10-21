# predict_hackathon.py
import argparse
import json
import os
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any, List, Optional
import time

import yaml
from hackathon_api import Datapoint, Protein, SmallMolecule

import torch
torch.set_float32_matmul_precision('high') #added to speed up evaluation

# --- filter thresholds & functions ---

import gemmi
SCORE_CACHE: dict[Path, float] = {}    # path -> score
DROP_CACHE: set[Path] = set()          # paths we already know to drop

# --- minimum contacts --- 

IFACE_CUTOFF = 5.0          # Å, heavy-atom distance to count as a contact
MIN_IFACE_CONTACTS = 80     # minimum contacts to accept a model

AA3 = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
    "SEC","PYL"
}

def _count_hl_to_a_contacts(pdb_path: Path, ab_chains=("H","L"), ag_chain="A", cutoff=IFACE_CUTOFF) -> int:
    """Count heavy-atom contacts between antibody chains (H/L) and antigen (A)."""
    st = gemmi.read_structure(str(pdb_path))
    st.remove_alternative_conformations()
    st.remove_hydrogens()

    ab_atoms = []
    ag_atoms = []
    for model in st:
        for chain in model:
            cname = chain.name
            if cname in ab_chains or cname == ag_chain:
                for res in chain:
                    if res.name not in AA3:
                        continue  # only protein residues
                    for atom in res:
                        if cname in ab_chains:
                            ab_atoms.append(atom)
                        elif cname == ag_chain:
                            ag_atoms.append(atom)

    if not ab_atoms or not ag_atoms:
        return 0

    cutoff2 = cutoff * cutoff
    contacts = 0
    for a in ab_atoms:
        ax, ay, az = a.pos.x, a.pos.y, a.pos.z
        for b in ag_atoms:
            dx = ax - b.pos.x
            dy = ay - b.pos.y
            dz = az - b.pos.z
            if (dx*dx + dy*dy + dz*dz) <= cutoff2:
                contacts += 1
    return contacts

# --- Minimum hydrogen bonds/salt bridges ---

# --- H-bond & salt bridge thresholds ---
HBOND_CUTOFF = 3.5          # Å, N···O (or O···N) heavy-atom distance
SALTBRIDGE_CUTOFF = 4.0     # Å, charged sidechain heavy-atom distance
MIN_HBONDS = 2              # minimal number of H-bonds across H/L <-> A
MIN_SALTBRIDGES = 0         # minimal number of salt bridges across H/L <-> A (it was too strict, restricted to 0 to act only as a bonus, not gate. Might re-evaluate later)

def _count_hbonds_and_saltbridges(pdb_path: Path, ab_chains=("H","L"), ag_chain="A",
                                  hbond_cutoff: float = HBOND_CUTOFF,
                                  sbridge_cutoff: float = SALTBRIDGE_CUTOFF) -> tuple[int, int]:
    """
    Count (H-bonds, salt-bridges) across Ab (H/L) and Ag (A) using geometry-only criteria.
    - H-bond proxy: any N···O or O···N ≤ hbond_cutoff Å between protein heavy atoms.
    - Salt-bridge: LYS/ARG (NZ/NH1/NH2) to ASP/GLU (OD1/OD2/OE1/OE2) ≤ sbridge_cutoff Å.
    """
    st = gemmi.read_structure(str(pdb_path))
    st.remove_alternative_conformations()
    st.remove_hydrogens()

    # Collect relevant atoms
    ab_N, ab_O = [], []
    ag_N, ag_O = [], []
    ab_pos = []   # positive groups for salt bridges (Lys/Arg)
    ag_pos = []
    ab_neg = []   # negative groups for salt bridges (Asp/Glu)
    ag_neg = []

    for model in st:
        for chain in model:
            cname = chain.name
            if cname not in (*ab_chains, ag_chain):
                continue
            for res in chain:
                rname = res.name
                if rname not in AA3:
                    continue  # protein only

                # Iterate atoms once; classify by element and role
                for atom in res:
                    aname = atom.name
                    elem = atom.element.name  # 'N', 'O', 'C', ...

                    # H-bond pools
                    if elem == "N":
                        if cname in ab_chains: ab_N.append(atom)
                        else:                  ag_N.append(atom)
                    elif elem == "O":
                        if cname in ab_chains: ab_O.append(atom)
                        else:                  ag_O.append(atom)

                    # Salt-bridge pools
                    # Positives: Lys NZ; Arg NH1/NH2/NE
                    if rname == "LYS" and aname == "NZ":
                        (ab_pos if cname in ab_chains else ag_pos).append(atom)
                    elif rname == "ARG" and aname in ("NH1","NH2"):
                        (ab_pos if cname in ab_chains else ag_pos).append(atom)

                    # Negatives: Asp OD1/OD2; Glu OE1/OE2
                    if rname == "ASP" and aname in ("OD1","OD2"):
                        (ab_neg if cname in ab_chains else ag_neg).append(atom)
                    elif rname == "GLU" and aname in ("OE1","OE2"):
                        (ab_neg if cname in ab_chains else ag_neg).append(atom)

    # Early exit
    if (not ab_N and not ab_O) or (not ag_N and not ag_O):
        return (0, 0)

    # Fast squared cutoff comparisons
    hb2 = hbond_cutoff * hbond_cutoff
    sb2 = sbridge_cutoff * sbridge_cutoff

    # H-bonds: (Ab N vs Ag O) + (Ab O vs Ag N)
    hbonds = 0
    for a in ab_N:
        ax, ay, az = a.pos.x, a.pos.y, a.pos.z
        for b in ag_O:
            dx = ax - b.pos.x; dy = ay - b.pos.y; dz = az - b.pos.z
            if (dx*dx + dy*dy + dz*dz) <= hb2:
                hbonds += 1
    for a in ab_O:
        ax, ay, az = a.pos.x, a.pos.y, a.pos.z
        for b in ag_N:
            dx = ax - b.pos.x; dy = ay - b.pos.y; dz = az - b.pos.z
            if (dx*dx + dy*dy + dz*dz) <= hb2:
                hbonds += 1

    # Salt bridges: (Ab + vs Ag -) + (Ab - vs Ag +)
    sbridges = 0
    for a in ab_pos:
        ax, ay, az = a.pos.x, a.pos.y, a.pos.z
        for b in ag_neg:
            dx = ax - b.pos.x; dy = ay - b.pos.y; dz = az - b.pos.z
            if (dx*dx + dy*dy + dz*dz) <= sb2:
                sbridges += 1
    for a in ab_neg:
        ax, ay, az = a.pos.x, a.pos.y, a.pos.z
        for b in ag_pos:
            dx = ax - b.pos.x; dy = ay - b.pos.y; dz = az - b.pos.z
            if (dx*dx + dy*dy + dz*dz) <= sb2:
                sbridges += 1

    return (hbonds, sbridges)
    
# --- Aromatic scoring (soft bonuses) ---
AROMATIC_PI_CUTOFF = 5.5   # Å between aromatic ring centroids across H/L <-> A
CATION_PI_CUTOFF   = 6.0   # Å from Lys/Arg cation N to aromatic ring centroid
W_PI_PI            = 3.0   # score bonus per π–π event
W_CATION_PI        = 4.0   # score bonus per cation–π event

def _collect_aromatic_ring_centers_and_cations(
    pdb_path: Path,
    ab_chains=("H","L"),
    ag_chain="A",
) -> tuple[list[gemmi.Position], list[gemmi.Position], list[gemmi.Atom], list[gemmi.Atom]]:
    """
    Returns (ab_ring_centers, ag_ring_centers, ab_cations, ag_cations).
    Aromatics: PHE/TYR phenyl (CG,CD1,CD2,CE1,CE2,CZ); TRP six-membered ring (CD2,CE2,CE3,CZ2,CZ3,CH2).
    Cations: LYS NZ; ARG NH1/NH2 (NE excluded intentionally).
    """
    st = gemmi.read_structure(str(pdb_path))
    st.remove_alternative_conformations()
    st.remove_hydrogens()

    ab_centers: list[gemmi.Position] = []
    ag_centers: list[gemmi.Position] = []
    ab_cations: list[gemmi.Atom]     = []
    ag_cations: list[gemmi.Atom]     = []

    phe_tyr = ("CG","CD1","CD2","CE1","CE2","CZ")
    trp6    = ("CD2","CE2","CE3","CZ2","CZ3","CH2")

    for model in st:
        for chain in model:
            cname = chain.name
            if cname not in (*ab_chains, ag_chain):
                continue
            role_ab = cname in ab_chains

            for res in chain:
                rname = res.name
                if rname not in AA3:
                    continue

                # cations
                if rname == "LYS":
                    nz = res.find_atom("NZ", '\0')
                    if nz:
                        (ab_cations if role_ab else ag_cations).append(nz)
                elif rname == "ARG":
                    for an in ("NH1","NH2"):
                        at = res.find_atom(an, '\0')
                        if at:
                            (ab_cations if role_ab else ag_cations).append(at)

                # ring centroids
                centers_target = ab_centers if role_ab else ag_centers
                if rname in ("PHE","TYR"):
                    pts = []
                    for an in phe_tyr:
                        at = res.find_atom(an, '\0')
                        if at:
                            pts.append(at.pos)
                    if len(pts) >= 5:
                        centers_target.append(gemmi.Position(
                            sum(p.x for p in pts)/len(pts),
                            sum(p.y for p in pts)/len(pts),
                            sum(p.z for p in pts)/len(pts),
                        ))
                elif rname == "TRP":
                    pts = []
                    for an in trp6:
                        at = res.find_atom(an, '\0')
                        if at:
                            pts.append(at.pos)
                    if len(pts) >= 5:
                        centers_target.append(gemmi.Position(
                            sum(p.x for p in pts)/len(pts),
                            sum(p.y for p in pts)/len(pts),
                            sum(p.z for p in pts)/len(pts),
                        ))

    return ab_centers, ag_centers, ab_cations, ag_cations


def _count_pi_pi_and_cation_pi(
    pdb_path: Path,
    ab_chains=("H","L"),
    ag_chain="A",
    pi_cutoff: float = AROMATIC_PI_CUTOFF,
    cpi_cutoff: float = CATION_PI_CUTOFF,
) -> tuple[int, int]:
    """Return (#π–π events, #cation–π events) across H/L <-> A using ring centroids and Lys/Arg terminal nitrogens."""
    ab_centers, ag_centers, ab_cations, ag_cations = _collect_aromatic_ring_centers_and_cations(
        pdb_path, ab_chains=ab_chains, ag_chain=ag_chain
    )
    pi_cnt = 0
    cpi_cnt = 0

    # π–π: centroid–centroid
    pi2 = pi_cutoff * pi_cutoff
    for p in ab_centers:
        for q in ag_centers:
            dx = p.x - q.x; dy = p.y - q.y; dz = p.z - q.z
            if (dx*dx + dy*dy + dz*dz) <= pi2:
                pi_cnt += 1

    # cation–π: cation N to centroid (both directions)
    cpi2 = cpi_cutoff * cpi_cutoff
    for n in ab_cations:
        nx, ny, nz = n.pos.x, n.pos.y, n.pos.z
        for c in ag_centers:
            dx = nx - c.x; dy = ny - c.y; dz = nz - c.z
            if (dx*dx + dy*dy + dz*dz) <= cpi2:
                cpi_cnt += 1
    for n in ag_cations:
        nx, ny, nz = n.pos.x, n.pos.y, n.pos.z
        for c in ab_centers:
            dx = nx - c.x; dy = ny - c.y; dz = nz - c.z
            if (dx*dx + dy*dy + dz*dz) <= cpi2:
                cpi_cnt += 1

    return pi_cnt, cpi_cnt

# --- Severe clash filter (inter-chain only) ---
CLASH_CUTOFF = 1.0          # Å considered a severe clash
MAX_SEVERE_CLASHES = 3      # drop if more than this many

def _count_severe_clashes(pdb_path: Path, ab_chains=("H","L"), ag_chain="A",
                          cutoff: float = CLASH_CUTOFF) -> int:
    """
    Count inter-chain severe clashes (H/L vs A): heavy-atom pairs < cutoff Å.
    Early-exits once the count exceeds MAX_SEVERE_CLASHES.
    """
    st = gemmi.read_structure(str(pdb_path))
    st.remove_alternative_conformations()
    st.remove_hydrogens()

    ab_atoms, ag_atoms = [], []
    for model in st:
        for chain in model:
            cname = chain.name
            if cname not in (*ab_chains, ag_chain):
                continue
            for res in chain:
                if res.name not in AA3:
                    continue
                for atom in res:
                    (ab_atoms if cname in ab_chains else ag_atoms).append(atom)

    if not ab_atoms or not ag_atoms:
        return 0

    c2 = cutoff * cutoff
    clashes = 0
    for a in ab_atoms:
        ax, ay, az = a.pos.x, a.pos.y, a.pos.z
        for b in ag_atoms:
            dx = ax - b.pos.x; dy = ay - b.pos.y; dz = az - b.pos.z
            if (dx*dx + dy*dy + dz*dz) < c2:
                clashes += 1
                if clashes > MAX_SEVERE_CLASHES:
                    return clashes
    return clashes

# --- Center-of-mass sanity (inter-chain separation) ---
MAX_CM_DISTANCE = 50.0   # Å; drop if Ab–Ag CM distance exceeds this

def _cm_distance_ab_ag(pdb_path: Path, ab_chains=("H","L"), ag_chain="A") -> float:
    """
    Return geometric-center distance (Å) between Ab (H/L) and Ag (A).
    Uses heavy atoms of standard amino acids; hydrogens removed.
    """
    st = gemmi.read_structure(str(pdb_path))
    st.remove_alternative_conformations()
    st.remove_hydrogens()

    ab_xyz = []
    ag_xyz = []
    for model in st:
        for chain in model:
            cname = chain.name
            if cname not in (*ab_chains, ag_chain):
                continue
            for res in chain:
                if res.name not in AA3:
                    continue
                for atom in res:
                    if cname in ab_chains:
                        ab_xyz.append(atom.pos)
                    else:
                        ag_xyz.append(atom.pos)

    if not ab_xyz or not ag_xyz:
        return float("inf")

    def _center(ps: list[gemmi.Position]) -> gemmi.Position:
        n = float(len(ps))
        return gemmi.Position(
            sum(p.x for p in ps)/n,
            sum(p.y for p in ps)/n,
            sum(p.z for p in ps)/n,
        )

    ca = _center(ab_xyz)
    cg = _center(ag_xyz)
    return ca.dist(cg)

# --- filter thresholds & functions ---


def prepare_protein_complex(datapoint_id: str, proteins: List[Protein], input_dict: dict, msa_dir: Optional[Path] = None) -> List[tuple[dict, List[str]]]:
    """
    Prepare input dict and CLI args for a protein complex prediction.
    You can return multiple configurations to run by returning a list of (input_dict, cli_args) tuples.
    Args:
        datapoint_id: The unique identifier for this datapoint
        proteins: List of protein sequences to predict as a complex
        input_dict: Prefilled input dict
        msa_dir: Directory containing MSA files (for computing relative paths)
    Returns:
        List of tuples of (final input dict that will get exported as YAML, list of CLI args). Each tuple represents a separate configuration to run.
    """
    # Please note:
    # `proteins`` will contain 3 chains
    # H,L: heavy and light chain of the Fv or Fab region
    # A: the antigen
    #
    # you can modify input_dict to change the input yaml file going into the prediction, e.g.
    # ```
    # input_dict["constraints"] = [{
    #   "contact": {
    #       "token1" : [CHAIN_ID, RES_IDX/ATOM_NAME], 
    #       "token1" : [CHAIN_ID, RES_IDX/ATOM_NAME]
    #   }
    # }]
    # ```
    #
    # will add contact constraints to the input_dict

    # Example: predict 5 structures
    cli_args = ["--diffusion_samples", "5"]
    return [(input_dict, cli_args)]

def prepare_protein_ligand(datapoint_id: str, protein: Protein, ligands: list[SmallMolecule], input_dict: dict, msa_dir: Optional[Path] = None) -> List[tuple[dict, List[str]]]:
    """
    Prepare input dict and CLI args for a protein-ligand prediction.
    You can return multiple configurations to run by returning a list of (input_dict, cli_args) tuples.
    Args:
        datapoint_id: The unique identifier for this datapoint
        protein: The protein sequence
        ligands: A list of a single small molecule ligand object 
        input_dict: Prefilled input dict
        msa_dir: Directory containing MSA files (for computing relative paths)
    Returns:
        List of tuples of (final input dict that will get exported as YAML, list of CLI args). Each tuple represents a separate configuration to run.
    """
    # Please note:
    # `protein` is a single-chain target protein sequence with id A
    # `ligands` contains a single small molecule ligand object with unknown binding sites
    # you can modify input_dict to change the input yaml file going into the prediction, e.g.
    # ```
    # input_dict["constraints"] = [{
    #   "contact": {
    #       "token1" : [CHAIN_ID, RES_IDX/ATOM_NAME], 
    #       "token1" : [CHAIN_ID, RES_IDX/ATOM_NAME]
    #   }
    # }]
    # ```
    #
    # will add contact constraints to the input_dict

    # Example: predict 5 structures
    cli_args = ["--diffusion_samples", "5"]
    return [(input_dict, cli_args)]

def post_process_protein_complex(datapoint: Datapoint,
                                 input_dicts: List[dict[str, Any]],
                                 cli_args_list: List[list[str]],
                                 prediction_dirs: List[Path]) -> List[Path]:
    """
    Filter by Ab–Ag interface size, then require minimal H-bonds and salt bridges,
    finally rank by a composite: contacts + 2*HB + 5*SB.
    """
    # 1) Collect PDBs once per call, avoiding duplicates within this call
    seen_in_call = set()
    all_pdbs: list[Path] = []
    for prediction_dir in prediction_dirs:
        for p in sorted(prediction_dir.glob(f"{datapoint.datapoint_id}_config_*_model_*.pdb")):
            if p not in seen_in_call:
                all_pdbs.append(p)
                seen_in_call.add(p)

    if not all_pdbs:
        return []

    ab_chains = ("H", "L")
    ag_chain = "A"

    scored: list[tuple[float, Path]] = []

    for pdb_path in all_pdbs:
        # 2) Fast exits using caches
        if pdb_path in DROP_CACHE:
            continue
        if pdb_path in SCORE_CACHE:
            scored.append((SCORE_CACHE[pdb_path], pdb_path))
            continue

        try:
            # (order: cheapest gates first)

            # Center-of-mass sanity (cheap)
            cm_dist = _cm_distance_ab_ag(pdb_path, ab_chains=ab_chains, ag_chain=ag_chain)
            if cm_dist > MAX_CM_DISTANCE:
                print(f"[post_process] Drop {pdb_path.name}: CM distance {cm_dist:.1f} Å > {MAX_CM_DISTANCE} Å")
                DROP_CACHE.add(pdb_path)
                continue

            # Severe clashes (still cheap)
            severe = _count_severe_clashes(pdb_path, ab_chains=ab_chains, ag_chain=ag_chain, cutoff=CLASH_CUTOFF)
            if severe > MAX_SEVERE_CLASHES:
                print(f"[post_process] Drop {pdb_path.name}: severe_clashes={severe} > {MAX_SEVERE_CLASHES}")
                DROP_CACHE.add(pdb_path)
                continue

            # Interface size
            n_contacts = _count_hl_to_a_contacts(pdb_path, ab_chains=ab_chains, ag_chain=ag_chain, cutoff=IFACE_CUTOFF)
            if n_contacts < MIN_IFACE_CONTACTS:
                print(f"[post_process] Drop {pdb_path.name}: contacts {n_contacts} < {MIN_IFACE_CONTACTS}")
                DROP_CACHE.add(pdb_path)
                continue

            # H-bonds & salt bridges
            hb, sb = _count_hbonds_and_saltbridges(
                pdb_path, ab_chains=ab_chains, ag_chain=ag_chain,
                hbond_cutoff=HBOND_CUTOFF, sbridge_cutoff=SALTBRIDGE_CUTOFF
            )
            if hb < MIN_HBONDS or sb < MIN_SALTBRIDGES:
                print(f"[post_process] Drop {pdb_path.name}: HB={hb} (<{MIN_HBONDS}) or SB={sb} (<{MIN_SALTBRIDGES})")
                DROP_CACHE.add(pdb_path)
                continue

            # Aromatic bonuses (soft)
            pi_pi, cpi = _count_pi_pi_and_cation_pi(
                pdb_path, ab_chains=ab_chains, ag_chain=ag_chain,
                pi_cutoff=AROMATIC_PI_CUTOFF, cpi_cutoff=CATION_PI_CUTOFF
            )

            # Composite score
            score = n_contacts + 2.0*hb + 5.0*sb + W_PI_PI*pi_pi + W_CATION_PI*cpi

            if score < 400:
                print(f"[post_process] Drop {pdb_path.name}: score={score:.1f} < 400 (too weak interface)")
                DROP_CACHE.add(pdb_path)
                continue

            print(f"[post_process] Keep {pdb_path.name}: contacts={n_contacts}, HB={hb}, SB={sb}, "
                  f"pi-pi={pi_pi}, cpi={cpi}, score={score:.1f}")

            SCORE_CACHE[pdb_path] = score             # cache keepers
            scored.append((score, pdb_path))

        except Exception as e:
            print(f"[post_process] Skipping {pdb_path.name} due to error: {e}")
            DROP_CACHE.add(pdb_path)

    if not scored:
        print("[post_process] WARNING: No models passed filters; using unfiltered models.")
        return all_pdbs

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored]


def post_process_protein_ligand(datapoint: Datapoint, input_dicts: List[dict[str, Any]], cli_args_list: List[list[str]], prediction_dirs: List[Path]) -> List[Path]:
    """
    Return ranked model files for protein-ligand submission.
    Args:
        datapoint: The original datapoint object
        input_dicts: List of input dictionaries used for predictions (one per config)
        cli_args_list: List of command line arguments used for predictions (one per config)
        prediction_dirs: List of directories containing prediction results (one per config)
    Returns: 
        Sorted pdb file paths that should be used as your submission.
    """
    # Collect all PDBs from all configurations
    all_pdbs = []
    for prediction_dir in prediction_dirs:
        config_pdbs = sorted(prediction_dir.glob(f"{datapoint.datapoint_id}_config_*_model_*.pdb"))
        all_pdbs.extend(config_pdbs)
    
    # Sort all PDBs and return their paths
    all_pdbs = sorted(all_pdbs)
    return all_pdbs

# -----------------------------------------------------------------------------
# ---- End of participant section ---------------------------------------------
# -----------------------------------------------------------------------------


DEFAULT_OUT_DIR = Path("predictions")
DEFAULT_SUBMISSION_DIR = Path("submission")
DEFAULT_INPUTS_DIR = Path("inputs")

ap = argparse.ArgumentParser(
    description="Hackathon scaffold for Boltz predictions",
    epilog="Examples:\n"
            "  Single datapoint: python predict_hackathon.py --input-json examples/specs/example_protein_ligand.json --msa-dir ./msa --submission-dir submission --intermediate-dir intermediate\n"
            "  Multiple datapoints: python predict_hackathon.py --input-jsonl examples/test_dataset.jsonl --msa-dir ./msa --submission-dir submission --intermediate-dir intermediate",
    formatter_class=argparse.RawDescriptionHelpFormatter
)

input_group = ap.add_mutually_exclusive_group(required=True)
input_group.add_argument("--input-json", type=str,
                        help="Path to JSON datapoint for a single datapoint")
input_group.add_argument("--input-jsonl", type=str,
                        help="Path to JSONL file with multiple datapoint definitions")

ap.add_argument("--msa-dir", type=Path,
                help="Directory containing MSA files (for computing relative paths in YAML)")
ap.add_argument("--submission-dir", type=Path, required=False, default=DEFAULT_SUBMISSION_DIR,
                help="Directory to place final submissions")
ap.add_argument("--intermediate-dir", type=Path, required=False, default=Path("hackathon_intermediate"),
                help="Directory to place generated input YAML files and predictions")
ap.add_argument("--group-id", type=str, required=False, default=None,
                help="Group ID to set for submission directory (sets group rw access if specified)")
ap.add_argument("--result-folder", type=Path, required=False, default=None,
                help="Directory to save evaluation results. If set, will automatically run evaluation after predictions.")

args = ap.parse_args()

def _prefill_input_dict(datapoint_id: str, proteins: Iterable[Protein], ligands: Optional[list[SmallMolecule]] = None, msa_dir: Optional[Path] = None) -> dict:
    """
    Prepare input dict for Boltz YAML.
    """
    seqs = []
    for p in proteins:
        if msa_dir and p.msa:
            if Path(p.msa).is_absolute():
                msa_full_path = Path(p.msa)
            else:
                msa_full_path = msa_dir / p.msa
            try:
                msa_relative_path = os.path.relpath(msa_full_path, Path.cwd())
            except ValueError:
                msa_relative_path = str(msa_full_path)
        else:
            msa_relative_path = p.msa
        entry = {
            "protein": {
                "id": p.id,
                "sequence": p.sequence,
                "msa": msa_relative_path
            }
        }
        seqs.append(entry)
    if ligands:
        def _format_ligand(ligand: SmallMolecule) -> dict:
            output =  {
                "ligand": {
                    "id": ligand.id,
                    "smiles": ligand.smiles
                }
            }
            return output
        
        for ligand in ligands:
            seqs.append(_format_ligand(ligand))
    doc = {
        "version": 1,
        "sequences": seqs,
    }
    return doc

def _run_boltz_and_collect(datapoint) -> None:
    """
    New flow: prepare input dict, write yaml, run boltz, post-process, copy submissions.
    """
    out_dir = args.intermediate_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    subdir = args.submission_dir / datapoint.datapoint_id
    subdir.mkdir(parents=True, exist_ok=True)

    # Prepare input dict and CLI args
    base_input_dict = _prefill_input_dict(datapoint.datapoint_id, datapoint.proteins, datapoint.ligands, args.msa_dir)

    if datapoint.task_type == "protein_complex":
        configs = prepare_protein_complex(datapoint.datapoint_id, datapoint.proteins, base_input_dict, args.msa_dir)
    elif datapoint.task_type == "protein_ligand":
        configs = prepare_protein_ligand(datapoint.datapoint_id, datapoint.proteins[0], datapoint.ligands, base_input_dict, args.msa_dir)
    else:
        raise ValueError(f"Unknown task_type: {datapoint.task_type}")

    # Run boltz for each configuration
    all_input_dicts = []
    all_cli_args = []
    all_pred_subfolders = []
    
    input_dir = args.intermediate_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    
    for config_idx, (input_dict, cli_args) in enumerate(configs):
        # Write input YAML with config index suffix
        yaml_path = input_dir / f"{datapoint.datapoint_id}_config_{config_idx}.yaml"
        with open(yaml_path, "w") as f:
            yaml.safe_dump(input_dict, f, sort_keys=False)

        # Run boltz
        cache = os.environ.get("BOLTZ_CACHE", str(Path.home() / ".boltz"))
        fixed = [
            "boltz", "predict", str(yaml_path),
            "--devices", "1",
            "--out_dir", str(out_dir),
            "--cache", cache,
            "--no_kernels",
            "--output_format", "pdb",
            "--seed", str(int(time.time() * 1e6) % 10_000_000)
        ]
        cmd = fixed + cli_args
        print(f"Running config {config_idx}:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

        # Compute prediction subfolder for this config
        pred_subfolder = out_dir / f"boltz_results_{datapoint.datapoint_id}_config_{config_idx}" / "predictions" / f"{datapoint.datapoint_id}_config_{config_idx}"
        
        all_input_dicts.append(input_dict)
        all_cli_args.append(cli_args)
        all_pred_subfolders.append(pred_subfolder)

    
    # Post-process and copy submissions
    # --- knobs (local; or set via env MIN_SUBMISSIONS / MAX_RERUNS) ---
    MIN_SUBMISSIONS = int(os.getenv("MIN_SUBMISSIONS", "5"))
    MAX_RERUNS = int(os.getenv("MAX_RERUNS", "3"))

    if datapoint.task_type == "protein_complex":
        ranked_files = post_process_protein_complex(datapoint, all_input_dicts, all_cli_args, all_pred_subfolders)
    elif datapoint.task_type == "protein_ligand":
        ranked_files = post_process_protein_ligand(datapoint, all_input_dicts, all_cli_args, all_pred_subfolders)
    else:
        raise ValueError(f"Unknown task_type: {datapoint.task_type}")

    # If we don't have enough passing models, run extra sampling rounds
    r = 0
    while len(ranked_files) < MIN_SUBMISSIONS and r < MAX_RERUNS:
        r += 1
        out_dir_round = args.intermediate_dir / "predictions" / f"round_{r}"
        out_dir_round.mkdir(parents=True, exist_ok=True)

        for config_idx, (input_dict, cli_args) in enumerate(configs):
            # re-use the YAMLs already written to input_dir
            yaml_path = (args.intermediate_dir / "input") / f"{datapoint.datapoint_id}_config_{config_idx}.yaml"
            cache = os.environ.get("BOLTZ_CACHE", str(Path.home() / ".boltz"))
            fixed = [
                "boltz", "predict", str(yaml_path),
                "--devices", "1",
                "--out_dir", str(out_dir_round),   # <-- round-specific folder
                "--cache", cache,
                "--no_kernels",
                "--output_format", "pdb",
                "--seed", str(int(time.time() * 1e7) % 10_000_000)
            ]
            cmd = fixed + cli_args
            print(f"Running round {r}, config {config_idx}:", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)

            pred_subfolder = (
                out_dir_round
                / f"boltz_results_{datapoint.datapoint_id}_config_{config_idx}"
                / "predictions"
                / f"{datapoint.datapoint_id}_config_{config_idx}"
            )
            all_input_dicts.append(input_dict)
            all_cli_args.append(cli_args)
            all_pred_subfolders.append(pred_subfolder)

        # re-run post-processing over ALL accumulated models
        if datapoint.task_type == "protein_complex":
            ranked_files = post_process_protein_complex(datapoint, all_input_dicts, all_cli_args, all_pred_subfolders)
        else:
            ranked_files = post_process_protein_ligand(datapoint, all_input_dicts, all_cli_args, all_pred_subfolders)

    if not ranked_files:
        raise FileNotFoundError(f"No model files found for {datapoint.datapoint_id}")

    # copy up to MIN_SUBMISSIONS (falls back to fewer if retries exhausted)
    top_n = min(len(ranked_files), MIN_SUBMISSIONS)
    for i, file_path in enumerate(ranked_files[:top_n]):
        target = subdir / (f"model_{i}.pdb" if file_path.suffix == ".pdb" else f"model_{i}{file_path.suffix}")
        shutil.copy2(file_path, target)
        print(f"Saved: {target}")

    if args.group_id:
        try:
            subprocess.run(["chgrp", "-R", args.group_id, str(subdir)], check=True)
            subprocess.run(["chmod", "-R", "g+rw", str(subdir)], check=True)
        except Exception as e:
            print(f"WARNING: Failed to set group ownership or permissions: {e}")

def _load_datapoint(path: Path):
    """Load JSON datapoint file."""
    with open(path) as f:
        return Datapoint.from_json(f.read())

def _run_evaluation(input_file: str, task_type: str, submission_dir: Path, result_folder: Path):
    """
    Run the appropriate evaluation script based on task type.
    
    Args:
        input_file: Path to the input JSON or JSONL file
        task_type: Either "protein_complex" or "protein_ligand"
        submission_dir: Directory containing prediction submissions
        result_folder: Directory to save evaluation results
    """
    script_dir = Path(__file__).parent
    
    if task_type == "protein_complex":
        eval_script = script_dir / "evaluate_abag.py"
        cmd = [
            "python", str(eval_script),
            "--dataset-file", input_file,
            "--submission-folder", str(submission_dir),
            "--result-folder", str(result_folder)
        ]
    elif task_type == "protein_ligand":
        eval_script = script_dir / "evaluate_asos.py"
        cmd = [
            "python", str(eval_script),
            "--dataset-file", input_file,
            "--submission-folder", str(submission_dir),
            "--result-folder", str(result_folder)
        ]
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    
    print(f"\n{'=' * 80}")
    print(f"Running evaluation for {task_type}...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")
    
    subprocess.run(cmd, check=True)
    print(f"\nEvaluation complete. Results saved to {result_folder}")

def _process_jsonl(jsonl_path: str, msa_dir: Optional[Path] = None):
    """Process multiple datapoints from a JSONL file."""
    print(f"Processing JSONL file: {jsonl_path}")

    for line_num, line in enumerate(Path(jsonl_path).read_text().splitlines(), 1):
        if not line.strip():
            continue

        print(f"\n--- Processing line {line_num} ---")

        try:
            datapoint = Datapoint.from_json(line)
            _run_boltz_and_collect(datapoint)

        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON on line {line_num}: {e}")
            continue
        except Exception as e:
            print(f"ERROR: Failed to process datapoint on line {line_num}: {e}")
            raise e
            continue

def _process_json(json_path: str, msa_dir: Optional[Path] = None):
    """Process a single datapoint from a JSON file."""
    print(f"Processing JSON file: {json_path}")

    try:
        datapoint = _load_datapoint(Path(json_path))
        _run_boltz_and_collect(datapoint)
    except Exception as e:
        print(f"ERROR: Failed to process datapoint: {e}")
        raise

def main():
    """Main entry point for the hackathon scaffold."""
    # Determine task type from first datapoint for evaluation
    task_type = None
    input_file = None
    
    if args.input_json:
        input_file = args.input_json
        _process_json(args.input_json, args.msa_dir)
        # Get task type from the single datapoint
        try:
            datapoint = _load_datapoint(Path(args.input_json))
            task_type = datapoint.task_type
        except Exception as e:
            print(f"WARNING: Could not determine task type: {e}")
    elif args.input_jsonl:
        input_file = args.input_jsonl
        _process_jsonl(args.input_jsonl, args.msa_dir)
        # Get task type from first datapoint in JSONL
        try:
            with open(args.input_jsonl) as f:
                first_line = f.readline().strip()
                if first_line:
                    first_datapoint = Datapoint.from_json(first_line)
                    task_type = first_datapoint.task_type
        except Exception as e:
            print(f"WARNING: Could not determine task type: {e}")
    
    # Run evaluation if result folder is specified and task type was determined
    if args.result_folder and task_type and input_file:
        try:
            _run_evaluation(input_file, task_type, args.submission_dir, args.result_folder)
        except Exception as e:
            print(f"WARNING: Evaluation failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

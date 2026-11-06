from modules.mode_spec import ModeSpec

MODE = ModeSpec(
    label="⚛️ Quantum mechanics",
    keywords=("schrodinger", "schrödinger", "wavefunction", "eigenstates", "tunneling", "tunnelling", "wavepacket", "qubit", "bloch", "commutator", "operator", "perturbation", "hamiltonian", "uncertainty", "bch", "density matrix", "decoherence", "t1", "t2", "particle in a box", "energy levels", "infinite well", "square well", "barrier"),
    module="ui.quantum", function="render_quantum_tab",
    needs_client=False,
    order=70,
)

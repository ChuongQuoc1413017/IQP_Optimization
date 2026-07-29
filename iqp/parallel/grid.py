INITS = ['normal', 'uniform', 'pi4', 'he', 'glorot']
HAMS  = ['ising', 'maxcut', 'partition']
ANSATZ_CONFIGS = [('iqp','single'), ('iqp','circular'), ('iqp','full'),
                  ('hea', 2), ('qaoa', 2)]
QUBITS    = [3, 6, 9, 12, 15]
N_PROBLEM = 50

def config_label(ansatz, spec):
    return spec if ansatz == 'iqp' else ansatz        # -> single/circular/full/hea/qaoa

def all_configs():
    cfgs = []
    for init in INITS:
        for ham in HAMS:
            for ansatz, spec in ANSATZ_CONFIGS:
                for n in QUBITS:
                    cfgs.append((init, ham, ansatz, spec, n))
    return cfgs

def shard_configs(shard, nshards):
    return [c for i, c in enumerate(all_configs()) if i % nshards == shard]

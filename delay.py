import os
from lava.utils.profiler import Profiler
from lava.proc.dense.process import Dense, DelayDense
from lava.proc.lif.process import LIF
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2HwCfg
import numpy as np
import argparse

# Number of time steps
TIME = 10_000_000

# Environment vars
os.environ["SLURM"] = "1"
os.environ["PARTITION"] = 'oheogulch_20m'
os.environ["BOARD"] = "ncl-ext-og-05"
os.environ["LOIHI_GEN"] = "N3B3"

# Create the parser
parser = argparse.ArgumentParser(description="Power profiling on Loihi 2")

# Add arguments
# Positional argument
parser.add_argument("--freq", type=int, choices=[25, 50, 100], default=25, help="Spiking frequency in Hz")
# Positional argument
parser.add_argument("--delay", type=int, choices=[0, 1, 5, 10, 15, 20], default=0, help="Synaptic Delay in ms")


# Parse arguments
args = parser.parse_args()
frequency = args.freq
delay = args.delay

if frequency == 100:
    bias_mant = 640
    bias_exp = 0
elif frequency == 50:
    bias_mant = 470
    bias_exp = 0
elif frequency == 25:
    bias_mant = 417
    bias_exp = 0



class Pair_lif():
    def __init__(self, bias_mant=0, bias_exp=0, delay=None):
        self.lif_src = LIF(shape=(1,), bias_mant=bias_mant,
                           bias_exp=bias_exp, vth=64, du=4095, dv=410)
        self.lif_dest = LIF(shape=(1,), bias_mant=0,
                            bias_exp=0, vth=64, du=4095, dv=4095)

        if delay == 0:
            self.dense = Dense(weights=np.ones((1, 1))*128)
        else:
            self.dense = DelayDense(weights=np.ones((1, 1))*128, delays=delay)
        
        # Connect the OutPort of lif_src to the InPort of dense.
        self.lif_src.s_out.connect(self.dense.s_in)
        # Connect the OutPort of dense to the InPort of lif_dst.
        self.dense.a_out.connect(self.lif_dest.a_in)

    def run(self, condition, run_cfg):
        self.lif_src.run(condition=condition, run_cfg=run_cfg)
    
    def stop(self):
        self.lif_src.stop()

run_config = Loihi2HwCfg()
results = {}
p = Pair_lif(bias_mant=bias_mant, bias_exp=bias_exp, delay=delay)
prof = Profiler.init(run_config)
# This can distort power consumption profiling. There is no need to record the time per timestep.
# Time can be computed from energy and power.
# prof.execution_time_probe(num_steps=TIME, buffer_size=1024,) 
prof.energy_probe(num_steps=TIME)
# prof.activity_probe()

# Execute Process lif_src and all Processes connected to it (dense, lif_dest).
p.run(condition=RunSteps(num_steps=TIME, blocking=True), run_cfg=run_config)
p.stop()

results["total_energy"], results["dynamic_energy"], results["static_energy"], results[
    "vdd_energy"], results["vdd_m_energy"], results["vdd_io_energy"] = prof.energy_breakdown()
print()
results["total_power"], results["dynamic_power"], results["static_power"], results[
    "vdd_power"], results["vdd_m_power"], results["vdd_io_power"] = prof.power_breakdown()
results["total_time"] = results["total_energy"]*10**-6/results["total_power"]
print()
print(f"Total execution time: {results['total_time']:.6f} s")


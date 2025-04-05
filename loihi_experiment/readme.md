# Loihi 2 experiment

## Environment
- **Python version**: 3.10.16
- **lava-nc version**: 0.10.0
- **Board**: Oheogultch ncl-ext-og-05

## Neuron Model
The neuron model follows the equation:

\[ \frac{dv}{dt} = \frac{I - v}{\tau} \]

### Parameters:
- **Threshold**: 1 (fixed-point value = 64)
- **Tau**: 10 ms (fixed-point value = 410)
- **Weight**: 1 (fixed-point value = 128)
- **Bias current (I)** varies to achieve different spiking frequencies:
  - **25 Hz**: mantissa = 417, exponent = 0
  - **50 Hz**: mantissa = 470, exponent = 0
  - **100 Hz**: mantissa = 640, exponent = 0

## Results Format
The results are structured as a nested dictionary:

1. **First key**: Frequency in Hz `["25", "50", "100"]`
2. **Second key**: Delay in ms `["0", "1", "5", "10", "15", "20"]`
3. **Third key**: Metrics
   - `dynamic_energy`
   - `static_energy`
   - `total_energy`
   - `dynamic_power`
   - `static_power`
   - `total_power`
   - `total_time`
4. **Units**:
   - Energies are in **Joules**
   - Powers are in **Watts**
   - Time is in **seconds**
   - All values correspond to a **simulation of 10⁷ iterations**
   - To obtain metrics **per iteration**, divide by **10⁷**
5. **Fourth key**: `"mean"` and `"std"` for 100 experiments per case

## Example Usage
To retrieve the **average ± std dynamic energy** for a spiking frequency of **25 Hz** with a **delay of 10 ms**:

```python
import json

with open("res.json", "r") as f:
    res = json.load(f)

print(f"{res['25']['10']['dynamic_energy']['mean']} ± {res['25']['10']['dynamic_energy']['std']}")
```



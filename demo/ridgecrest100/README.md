# Ridgecrest demo inputs

Ready-to-run inputs for the "Complete demo command sequence" in the
repository README: 100 events from the 2019-07-04 to 07-09 Ridgecrest
sequence (spanning the M6.4 and M7.1 mainshocks), each with at least 12 P
picks — 2,222 picks in total.

| File | Content |
|---|---|
| `picks.csv` | P pick table: `waveform_path,p_time,record_id,event_id,station` |
| `events.csv` | event catalog: `event_id,lat,lon,dep,mag,time` |
| `stations.csv` | station list: `station(NET.STA),lat,lon,ele` |
| `vel.txt` | 1-D velocity model for SKHASH |
| `waveforms/*.sac` | one Z-component trace per pick, P ± 45 s (SAC header: `a` = 45 s; station/event coordinates in `stla/stlo/…/evla/evlo/evdp`) |

Data sources: continuous waveforms from the SCEDC FDSN service (CI network,
Southern California Seismic Network), cut around PhaseNet consensus picks
from the paper's Ridgecrest data pipeline. To run the demo:

```bash
cd demo/ridgecrest100
# then run the four commands from the README "Complete demo command sequence"
```

Waveform data © SCEDC (https://service.scedc.scsn.org, doi:10.7909/C3WD3xH1),
used here for demonstration.

# push-pull-anomaly-tracking

Code implementing the performance evaluation of the paper "A Combined Push-Pull Access Framework for Digital Twin Alignment and Anomaly Reporting"

## Abstract
> A digital twin (DT) contains a set of virtual models of real systems and processes that are synchronized to their physical counterparts. This enables experimentation and examination of counterfactuals, simulating the consequences of decisions in real time. However, the DT accuracy relies on timely updates that maintain alignment with the real system. We can distinguish between: (i) pull-updates, which follow a request from the DT to the sensors, to decrease its drift from the physical state; (ii) push-updates, which are sent directly by the sensors since they represent urgent information, such as anomalies. In this work, we devise a push-pull scheduler (PPS) medium access framework, which dynamically allocates the communication resources used for these two types of updates. Our scheme strikes a balance in the trade-off between DT alignment in normal conditions and anomaly reporting, optimizing resource usage and reducing the drift age of incorrect information (AoII) by over 20% with respect to state-of-the-art solutions, while maintaining the same anomaly detection guarantees, as well as reducing the worst-case anomaly detection AoII from 70 ms to 20 ms when considering a 1 ms average drift AoII constraint.

The paper is submitted to [IEEE INFOCOM 2026](https://infocom2026.ieee-infocom.org/group/81).
An arxiv version is [MISSING FOR NOW](missing)

The main performance are obtainable by running the scripts having ``_analysis`` at the end of the name, so, e.g.,
```
pull_frame_analysis.py
push_frame_analysis.py
coexistence_frame_analysis.py
```
will perform the simulation varying the available resources for pull-only, push-only, and push-pull scenario.

Note that the scripts should be run with the flag ```--override``` to overwrite the ``.csv`` results into the ``data`` folder. Alternatively, you can use the ``--savedir [path/to/folder/]`` to save the data in a custom folder.

The standard parameters can be edited changing ```common.py```. 
The schedulers are implemented in ``pull_scheduler.py``, ``push_scheduler.py``, while the resource managers are in ``push_pull_managers.py``.

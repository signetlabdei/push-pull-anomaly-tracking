# push-pull-anomaly-tracking

Code implementing the performance evaluation of papers "A Combined Push-Pull Access Framework for Digital Twin Alignment and Anomaly Reporting" and "Push-Pull Medium Access for Digital Twin Alignment and Low-Latency Anomaly Reporting"

## Abstract of "A Combined Push-Pull Access Framework for Digital Twin Alignment and Anomaly Reporting"
> A digital twin (DT) contains a set of virtual models of real systems and processes that are synchronized to their physical counterparts. This enables experimentation and examination of counterfactuals, simulating the consequences of decisions in real time. However, the DT accuracy relies on timely updates that maintain alignment with the real system. We can distinguish between: (i) pull-updates, which follow a request from the DT to the sensors, to decrease its drift from the physical state; (ii) push-updates, which are sent directly by the sensors since they represent urgent information, such as anomalies. In this work, we devise a push-pull scheduler (PPS) medium access framework, which dynamically allocates the communication resources used for these two types of updates. Our scheme strikes a balance in the trade-off between DT alignment in normal conditions and anomaly reporting, optimizing resource usage and reducing the drift age of incorrect information (AoII) by over 20% with respect to state-of-the-art solutions, while maintaining the same anomaly detection guarantees, as well as reducing the worst-case anomaly detection AoII from 70 ms to 20 ms when considering a 1 ms average drift AoII constraint.

The paper has been accepted to and presented in [IEEE INFOCOM ASoI 2026](https://infocom2026.ieee-infocom.org/group/81).
A preprint version is available on [arXiv](https://arxiv.org/abs/2508.21516)

## Abstract of "Push-Pull Medium Access for Digital Twin Alignment and Low-Latency Anomaly Reporting"
> A digital twin (DT) contains a set of virtual models of real systems and processes that are synchronized with their physical counterparts. In a setup in which contact with the physical world is maintained through sensors and actuators that are wirelessly connected to the DT’s computing engine, DT alignment requires periodic status updates, while safety-critical messages and fault conditions call for low-latency anomaly reporting, creating a fundamental trade-off in how wireless resources are used. We present a medium access framework combining pull-based updates, centrally scheduled according to goal-oriented principles, with urgent push-based updates, for which transmission decisions are made directly by the sensors. This enables the system to quickly detect and recover from anomalies while maintaining DT alignment. We thus design a push-pull scheduler (PPS) that strikes a balance in the trade-off between DT alignment in normal conditions and anomaly reporting, optimizing resource usage and reducing DT drift by 20 − 30% with respect to state-of-the-art solutions while maintaining the same anomaly detection guarantees, or reducing worst-case anomaly detection times by 30 − 70% while meeting the same DT alignment conditions

The paper has been submitted to IEEE Journal of Selected Areas of Communication.
A preprint version is available on [arXiv](https://arxiv.org/abs/)


## Usage
The main performance results and figures are obtainable by running the scripts having ``_analysis`` at the end of the name, so, e.g.,
```
pull_frame_analysis.py
push_frame_analysis.py
coexistence_frame_analysis.py
```
will perform the simulation varying the available resources for pull-only, push-only, and push-pull scenario.
Using the ``--parallel`` flag will run the different episodes in parallel.

Note that the scripts should be run with the flag ```--overwrite``` to overwrite the ``.csv`` results into the ``data`` folder. Alternatively, you can use the ``--savedir [path/to/folder/]`` to save the data in a custom folder.

The standard parameters can be edited changing ```common.py```. 
The schedulers are implemented in ``pull_scheduler.py``, ``pull_kalman_scheduler.py``, ``push_scheduler.py``, while the resource managers are in ``push_pull_managers.py``. In general, if ``*_kalman*`` is in the name, the script refer to the continuous tracking task using a Kalman filter to track DT drifts, while the binary HMM is assumed otherwise.

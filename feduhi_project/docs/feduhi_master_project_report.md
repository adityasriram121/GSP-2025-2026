# A Revolutionary IT Science Fair Project: Mitigating Urban Heat Islands with a Privacy-Preserving Federated Learning Network on Low-Cost Edge Devices

## Executive Summary

This master-level science fair project blueprint addresses the Urban Heat Island (UHI) effect by pioneering the **Federated Urban Heat Island (FedUHI)** model. The solution combines **Federated Learning (FL)** and **Edge Computing (EC)** on a low-cost Raspberry Pi cluster to predict temperature hotspots in real time while protecting individual privacy. The document provides a complete roadmap—from theory and experimental design to hardware, software, and evaluation—so that a disciplined, technically inclined student can implement a project that aligns with **United Nations Sustainable Development Goals (SDGs) 11 & 13**.

---

## Part I · The Problem and the Paradigm Shift

### Chapter 1 · The Urban Heat Island Effect and the Data Gap

- **Urban Heat Island (UHI)**: Cities absorb and retain more heat than rural areas due to dense infrastructure, limited vegetation, and anthropogenic heat sources. Vulnerable populations experience disproportionate risk during extreme heat events.
- **Global Relevance**: UHI mitigation supports SDG 11 (Sustainable Cities and Communities) and SDG 13 (Climate Action) by reducing environmental impacts and enhancing climate resilience.
- **Traditional Approaches**: Tree planting, green roofs, and reflective pavements mitigate heat but rely on coarse data and reactive interventions.
- **Data Challenge**:
  - **Privacy**: Centralizing fine-grained sensor data risks revealing personal schedules, energy usage, and movement patterns.
  - **Bandwidth**: IoT devices struggle to stream continuous high-resolution data to the cloud.
  - **Latency**: Cloud-centric pipelines introduce delays that hinder real-time response during heat emergencies.

### Chapter 2 · Introducing Federated Edge Learning: The Revolutionary Solution

- **Edge Computing (EC)**: Processes data near the source to reduce latency, conserve bandwidth, and improve security.
- **Federated Learning (FL)**: Trains a shared model across decentralized clients that never expose raw data, following the classic **Federated Averaging (FedAvg)** cycle:
  1. Server distributes an initial model.
  2. Clients train locally on private datasets.
  3. Clients send weight updates (not raw data).
  4. Server aggregates updates and redistributes the improved model.
- **Combined Paradigm**: Each Raspberry Pi node hosts sensors, trains locally, and contributes updates to the global FedUHI model. This approach transforms privacy, bandwidth, and latency constraints into strengths by distributing computation while protecting data.

---

## Part II · The Scientific Method in Action: A Master-Level Experiment

### Chapter 3 · Hypothesis, Variables, and Experimental Design

- **Hypothesis**: Training a model on a low-cost edge network using federated learning will achieve predictive accuracy comparable to centralized training while reducing network bandwidth usage and preserving privacy.
- **Null Hypothesis**: The federated approach will perform significantly worse in accuracy, training time, or communication overhead.
- **Variables**:
  - **Independent Variable**: Learning paradigm (Federated Learning vs. Simulated Centralized Learning).
  - **Dependent Variables**: Model accuracy, bandwidth consumption, training time.
  - **Controlled Variables**: Hardware platform, model architecture, dataset, and data collection rate.
- **Experimental Phases**:
  1. **Centralized Baseline**: Aggregate all zone data on one node, train the baseline model, measure accuracy, training time, and network traffic.
  2. **Federated Experiment**: Reconfigure the cluster for federated learning, keep data local, exchange model updates, and measure the same metrics.
  3. **Comparative Analysis**: Evaluate how the federated model performs relative to centralized training, highlighting trade-offs and advantages.
- **Variable Table**:

| Variable Type      | Variable               | Definition / Value                                              |
|--------------------|------------------------|------------------------------------------------------------------|
| Independent        | Learning Paradigm      | (A) Federated Learning, (B) Simulated Centralized Learning      |
| Dependent          | Model Accuracy         | Percentage of correct temperature hotspot predictions           |
| Dependent          | Bandwidth Consumption  | Total data transferred during training (MB/GB)                  |
| Dependent          | Training Time          | Time to reach convergence (minutes/seconds)                     |
| Controlled         | Hardware Platform      | 4-node Raspberry Pi 5 cluster                                   |
| Controlled         | Model Architecture     | Shared neural network topology and hyperparameters              |
| Controlled         | Dataset                | Identical synthetic or real sensor readings                      |
| Controlled         | Data Collection Rate   | Fixed sampling cadence (e.g., every 60 seconds)                 |

### Chapter 4 · Experimental Procedure

1. **Centralized Run**:
   - Aggregate temperature and humidity data from all nodes on a central server.
   - Train the global model and log accuracy, training duration, and total bytes transferred.
2. **Federated Run**:
   - Deploy the Flower-based FL server on the coordinator node.
   - Allow each Raspberry Pi to train locally and send weight updates only.
   - Measure accuracy, convergence behavior, per-round bandwidth usage, and total training time.
3. **Analysis**:
   - Compare metrics across paradigms.
   - Discuss privacy guarantees, bandwidth savings, and responsiveness improvements.
   - Document lessons learned, confounding factors, and next steps.

---

## Part III · From Concept to Concrete Reality: Building the Project

### Chapter 5 · Hardware: A Low-Cost Edge Network

The project uses a tangible Raspberry Pi cluster to make the edge + federated paradigm accessible.

| Component                     | Quantity | Unit Cost (USD) | Total Cost (USD) |
|-------------------------------|----------|-----------------|------------------|
| Raspberry Pi 5 (8GB/16GB)     | 4        | $76–$93         | $304–$372        |
| MicroSD Card (32GB+)          | 4        | $13             | $52              |
| Temperature/Humidity Sensor   | 4        | $5              | $20              |
| Network Switch (PoE optional) | 1        | $50             | $50              |
| Power Supply                  | 4        | $20             | $80              |
| Cluster Case/Rack             | 1        | $20–$65         | $20–$65          |
| Cables & Miscellaneous        | 1 set    | $15             | $15              |
| **Estimated Total**           | —        | —               | **$541–$654**    |

### Chapter 6 · Software: Frameworks and Implementation

- **Data Collection**: Python scripts on each Pi collect sensor readings (temperature, humidity, timestamp) at a fixed interval.
- **Local Training**: Lightweight neural networks train on-device with configurable epochs, batch size, and normalization.
- **Federated Coordination**: The [Flower](https://flower.ai) framework orchestrates model distribution, update aggregation, and performance tracking.
- **Non-IID Awareness**: Incorporate normalization and monitoring strategies to handle heterogeneous microclimates (sunny vs. shaded zones).
- **Ethical Considerations**: Observe potential bias introduced by uneven data distribution and document mitigation ideas.

---

## Part IV · Measuring Impact and Future Directions

### Chapter 7 · Metrics of Success

Collect metrics for both paradigms and summarize improvements:

| Metric                  | Federated (FL) | Centralized (CL) | Improvement / Change (%)                |
|-------------------------|----------------|------------------|-----------------------------------------|
| Model Accuracy          | A<sub>FL</sub> | A<sub>CL</sub>   | `(A_FL - A_CL) / A_CL × 100`             |
| Training Time           | T<sub>FL</sub> | T<sub>CL</sub>   | `(T_CL - T_FL) / T_CL × 100`             |
| Bandwidth Consumption   | B<sub>FL</sub> | B<sub>CL</sub>   | `(B_CL - B_FL) / B_CL × 100`             |

Use the pipeline’s CSV summaries and plots to support quantitative findings, and narrate qualitative observations about privacy, latency, and resiliency.

### Chapter 8 · Conclusion and Expansion

- **Proof of Concept**: Demonstrate that a low-cost edge network can support privacy-preserving collaborative temperature prediction.
- **Scalability**: Extend the network with more nodes and diverse locations to improve generalization.
- **Sensor Fusion**: Incorporate air quality, light, and noise sensors to broaden environmental insights.
- **Algorithmic Enhancements**: Experiment with advanced FL strategies tailored for non-IID data and fairness.
- **Actionable Dashboards**: Create real-time visualizations and alerts for city planners and communities.

---

## Works Cited

1. Reduce Heat Islands | US EPA. https://www.epa.gov/green-infrastructure/reduce-heat-islands
2. Heat Island Reduction Solutions | US EPA. https://www.epa.gov/heatislands/heat-island-reduction-solutions
3. Goal 11 | Department of Economic and Social Affairs. https://sdgs.un.org/goals/goal11
4. Sustainable Development Goal 13. https://en.wikipedia.org/wiki/Sustainable_Development_Goal_13
5. Sustainable Mitigation Strategies for Urban Heat Island Effects in Urban Areas. https://www.mdpi.com/2071-1050/15/14/10767
6. Cooling Cities Strategies and Technologies to Mitigate Urban Heat. https://climate-adapt.eea.europa.eu/en/metadata/publications/cooling-cities-strategies-and-technologies-to-mitigate-urban-heat
7. The Role of Federated Learning in Smart Cities. https://www.francescatabor.com/articles/2025/6/16/the-role-of-federated-learning-in-smart-cities-enhancing-traffic-parking-and-environmental-monitoring-with-privacy-preserving-ai
8. Introduction to Federated Learning | Kuan Hoong. https://kuanhoong.medium.com/introduction-to-federated-learning-cd4cf6e9a0b9
9. Federated Learning in Machine Learning Pipelines - Meegle. https://www.meegle.com/en_us/topics/federated-learning/federated-learning-in-machine-learning-pipelines
10. Top 9 Interesting Edge Computing Research Topics. https://www.phddirection.com/edge-computing-research-topics/
11. Federated Learning for IoT: A Survey of Techniques, Challenges, and Applications. https://www.mdpi.com/2224-2708/14/1/9
12. Powering the Future: Edge Computing in Smart Cities - SNUC. https://snuc.com/blog/edge-computing-smart-cities/
13. Federated Learning in Edge Computing: A Systematic Survey. https://pmc.ncbi.nlm.nih.gov/articles/PMC8780479/
14. Edge and Cloud Computing in Smart Cities - MDPI. https://www.mdpi.com/1999-5903/17/3/118
15. Sustainable edge computing: Challenges and future directions. https://www.researchgate.net/publication/384610192_Sustainable_edge_computing_Challenges_and_future_directions
16. Smart City Solutions – Intel. https://www.intel.com/content/www/us/en/internet-of-things/smart-cities.html
17. How is federated learning applied in remote sensing? - Milvus. https://milvus.io/ai-quick-reference/how-is-federated-learning-applied-in-remote-sensing
18. What is Federated Learning? - Flower Framework. https://flower.ai/docs/framework/tutorial-series-what-is-federated-learning.html
19. Federated Learning for IoT Devices with Domain Generalization. https://pages.mtu.edu/~xinyulei/Papers/IOTJ22.pdf
20. OpenFL – Linux Foundation Project. https://openfl.io/
21. Hypotheses - The University Writing Center. https://writingcenter.tamu.edu/writing-speaking-guides/hypotheses
22. My Proven Method For Writing An Experiment Hypothesis. https://www.firstprinciples.ventures/insights/my-proven-method-for-writing-an-experiment-hypothesis
23. Experimental Design in Data Science - ScholarHat. https://www.scholarhat.com/tutorial/datascience/experimental-design-in-data-science
24. Variables in Machine Learning - EITC. http://eitc.org/research-opportunities/new-media-and-new-digital-economy/data-science-and-analytics/foundations-of-data-science-and-analytics/feature-engineering/variables-in-machine-learning
25. Independent and Dependent Variables: Definitions and Differences - Great Learning. https://www.mygreatlearning.com/blog/independent-and-dependent-variables/
26. Managing Machine Learning Experiments - Medium. https://medium.com/data-science/a-quick-guide-to-managing-machine-learning-experiments-af84da6b060b
27. Edge Computing Solutions For Enterprise - NVIDIA. https://www.nvidia.com/en-us/edge-computing/
28. IoT Edge Computing | Dell. https://www.dell.com/en-us/shopping/iot-edge-computing
29. Building a Raspberry Pi Cluster: Step-by-Step Guide. https://www.sunfounder.com/blogs/news/building-a-raspberry-pi-cluster-step-by-step-guide-and-practical-applications
30. Turing Pi - Buy Cluster on a mini ITX board with Raspberry Pi. https://turingpi.com/
31. Raspberry Pi Cluster - Walmart. https://www.walmart.com/c/kp/raspberry-pi-cluster
32. Raspberry Pi Cluster - Etsy. https://www.etsy.com/market/raspberry_pi_cluster
33. Open-source Tools for Federated Learning - Milvus. https://milvus.io/ai-quick-reference/what-are-some-opensource-tools-for-federated-learning
34. Impact of Non-IID Data in Federated Learning - Milvus. https://milvus.io/ai-quick-reference/what-is-the-impact-of-noniid-data-in-federated-learning
35. Distribution-Regularized Federated Learning on Non-IID Data - Zimu Zhou. https://zhouzimu.github.io/paper/icde23-wang.pdf
36. Addressing Bias and Fairness Using Fair Federated Learning - MDPI. https://www.mdpi.com/2079-9292/13/23/4664
37. Mitigating Group Bias in Federated Learning - Cisco Outshift. https://outshift.cisco.com/blog/mitigating-group-bias-federated-learning-beyond-local-fairness
38. AI Enhancing Smart Cities and Urban Living - Stack AI. https://www.stack-ai.com/blog/how-is-ai-enhancing-smart-cities-and-urban-living
39. AI in Smart Cities - Dell Technologies. https://www.delltechnologies.com/asset/en-us/solutions/industry-solutions/industry-market/ai-for-digital-cities-whitepaper.pdf

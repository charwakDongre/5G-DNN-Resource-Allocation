# 5G RESOURCE ALLOCATION USING DEEP NEURAL NETWORKS

## 🌟 Project Overview

This project implements a complete **intelligent Radio Resource Management (RRM) framework** for Fifth-Generation (5G) networks. It utilizes a **Deep Neural Network (DNN)** classifier to dynamically map heterogeneous application traffic to one of three core 5G network slices—**eMBB, URLLC, and mMTC**—to ensure differentiated Quality of Service (QoS) guarantees.

The solution moves beyond static allocation by integrating the DNN agent into a detailed system-level simulation that models 5G physical layer constraints, including Link Adaptation (LA) and a realistic HARQ-based Packet Loss Rate (PLR) mechanism.

## 🎯 Key Objectives

1.  **High-Accuracy Classification:** Train a DNN to classify network QoS metrics into 9 application types with high accuracy ($\mathbf{> 95\%}$).
2.  **Service Differentiation:** Successfully enforce the conflicting QoS requirements of $\text{eMBB}$ (high throughput), $\text{URLLC}$ (extreme reliability), and $\text{mMTC}$ (massive connectivity).
3.  **Address Starvation:** Implement a **Cost-Sensitive Loss function** to ensure critical, low-frequency $\text{URLLC}$ traffic is prioritized and not starved of resources.

## 💻 Methodology Summary

The project follows a four-phase structure:

1.  **Data Preparation:** Raw QoS data is standardized and split into Train/Validation/Test sets.
2.  **DNN Training:** A Multi-Layer Perceptron (MLP) with two hidden layers ($\text{64}$ / $\mathbf{256}$ neurons) is trained using the **Adam optimizer** and a piecewise learning rate scheduler.
3.  **5G Simulation:** The trained DNN is integrated into a slot-by-slot MATLAB simulator. The DNN's output (Application Type) is mapped to a predefined **Network Slice**, triggering the appropriate resource provisioning (e.g., power, bandwidth, $\text{MCS}$).
4.  **PLR Modeling:** Reliability is determined using a physical layer model that incorporates HARQ retransmissions and a non-negotiable **System PLR Floor** to enforce slice-specific QoS guarantees (e.g., $10^{-5}$ for $\text{URLLC}$).

## 💡 Key Results

The simulation successfully validates the framework's ability to enforce strict 5G performance trade-offs:

| Metric | eMBB (Throughput) | URLLC (Reliability) | mMTC (Massive IoT) |
| :--- | :--- | :--- | :--- |
| **Average GBR** | $\mathbf{40.00 \text{ Mbps}}$ (Highest) | $2.00 \text{ Mbps}$ (Lowest) | $1.00 \text{ Mbps}$ |
| **Average PLR** | $\mathbf{0.00\%}$ | $\mathbf{0.00\%}$ (Meets $99.999\%$ Target) | $4.16\%$ (Highest Loss Tolerance) |
| **Classification Acc.** | \multicolumn{3}{|c|}{$\mathbf{95.00\%}$} |
| **Core Finding** | Achieves max throughput under best-effort latency. | Demonstrates successful enforcement of the **Latency Bound**. | Conserves resources with acceptable loss tolerance. |

## 🛠️ Technologies Used

* **Platform:** MATLAB (Deep Learning Toolbox, Simulation Environment)
* **Model:** Deep Neural Network (Multi-Layer Perceptron)
* **Optimization:** Adam Optimizer, Cost-Sensitive Loss Function
* **Networking:** 5G New Radio ($\text{NR}$) Slicing (eMBB/URLLC/mMTC), Link Adaptation, HARQ Protocol Modeling

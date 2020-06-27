# Towards Robust Ensemble Defense Against Adversarial Examples Attack
Research has shown that many state-of-the-art models are vulnerable to attacks by well-crafted adversarial examples. The exposed vulnerabilities of these models raise the question of its usability in safety-critical real-world applications such as autonomous driving and applications in the field of medicine. We present a new ensemble defense strategy using adversarial retraining technique which is capable of withstanding six adversarial attacks on cifar10 dataset with a minimum accuracy of 89.31%. Details of the experiment setup and results were presented in *Globecom 2019* and is published as conference proceedings.

Here is the link to the paper: [Towards Robust Ensemble Defense Against Adversarial Examples Attack](https://ieeexplore.ieee.org/abstract/document/9013408)

Attacks Implimented:

    Fast Gradient Sign Method (FGSM)
    Basic Iterative Method (BIM)
    Iterative Least-Likely Class (ILLC)
    DeepFool
    Carlini-Wagner (CW) L2
    Carlini-Wagner (CW) Linf
    
Defense Method:

    Ensemble of retrained models with soft voting strategy


If you would like to use and extend my code/research please cite the work as:

**N. Mani, M. Moh and T. Moh, "Towards Robust Ensemble Defense Against Adversarial Examples Attack," 2019 IEEE Global Communications Conference (GLOBECOM),     Waikoloa, HI, USA, 2019, pp. 1-6, doi: 10.1109/GLOBECOM38437.2019.9013408**

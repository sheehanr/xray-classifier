# X-Ray Classifier

## Key Features

* Classify X-Rays by anatomical region into 19 classes including "Other" for less common body parts.
* Ability to handle inputs of JPEG, PNG, and DICOM files.
* Maintain accuracy of over 95% in identifying anatomical region of X-Rays 

---

## Model Engineering

**Goal:** Classify X-Ray imaging files anatomical region with >95% accuracy, with graceful handling of unrecognized imaging.

* **Base Model:** ResNet18
* **Training:** Transfer learning on 40,000+ imaging studies using PyTorch and Torchvision.
  * Used an NVIDIA RTX 5080 GPU to speed up training. 
* **Data Balancing:** Undersampled large datasets at 5,000 images and applied weighted loss techniques to smaller datasets.
* **Target Classes:** Classifies the following 19 specific clinical views:
  * **X-Ray:** Ankle, Chest, Elbow, Finger, Foot, Forearm, Hand, Hip, Humerus, Knee, Shoulder, Wrist, Other

---

## Tech Stack

* **Language:** Python 3.12
* **Python Libraries:** Kaggle, NumPy, Pandas, Pillow, PyDicom, TQDM, PyTorch, Torchvision
* **Hardware:** NVIDIA RTX 5080 (Training)

---

## Training Data & Acknowledgements

All imaging study datasets were sourced from [Kaggle](https://www.kaggle.com). 

* **[NIH Chest X-rays](https://www.kaggle.com/datasets/nih-chest-xrays/data)** by National Institutes of Health Chest X-Ray Dataset
  * Used for: X-Ray Chest
  * Original Source: National Institutes of Health (NIH)
  * Citation: Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. IEEE CVPR 2017
  * License: [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)

* **[mura v11](https://www.kaggle.com/datasets/cjinny/mura-v11)** by Jincui
  * Used for: X-Ray Elbow, X-Ray Finger, X-Ray Forearm, X-Ray Hand, X-Ray Humerus, X-Ray Shoulder, X-Ray Wrist
  * Original Source: Stanford Center for Artificial Intelligence in Medicine & Imaging (AIMI)
  * Citation: [MURA: MSK Xrays](https://doi.org/10.71718/cwh3-0p32)
  * License: Unknown

* **[The UNIFESP X-Ray Body Part Classification Dataset](https://www.kaggle.com/datasets/felipekitamura/unifesp-xray-bodypart-classification)** by FelipeKitamura, MD, PhD & Eduardo Farina
  * Used for: X-Ray Abdomen, X-Ray Ankle, X-Ray Cervical Spine, X-Ray Chest, X-Ray Clavicles, X-Ray Elbow, X-Ray Feet, X-Ray Finger, X-Ray Forearm, X-Ray Hand, X-Ray Hip, X-Ray Knee, X-Ray Lower Leg, X-Ray Lumbar Spine, X-Ray Others, X-Ray Pelvis, X-Ray Shoulder, X-Ray Sinus, X-Ray Skull, X-Ray Thigh, X-Ray Thoracic Spine, X-Ray Wrist
  * Original Source: Universidade Federal de São Paulo (UNIFESP)
  * Citation: Eduardo Moreno Judice de Mattos Farina, Nitamar Abdala, and Felipe Campos Kitamura. (2022). The UNIFESP X-Ray Body Part Classification Dataset [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/3399135
  * License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

* **[Digital Knee X-ray Images](https://www.kaggle.com/datasets/orvile/digital-knee-x-ray-images)** by Orvile
  * Used for: X-Ray Knee
  * Original Source: Mendeley Data
  * Citation: Gornale, Shivanand; Patravali, Pooja (2020), Digital Knee X-ray Images, Mendeley Data, V1, https://doi.org/10.17632/t9ndx37v5h.1
  * License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

* **[Multi-Class Knee Osteoporosis X-Ray Dataset](https://www.kaggle.com/datasets/mohamedgobara/multi-class-knee-osteoporosis-x-ray-dataset)** by Mohamed Gobara
  * Used for: X-Ray Knee
  * Original Source: Knee Osteoporosis Diagnosis Based on Deep Learning
  * Citation: Sarhan, A.M., Gobara, M., Yasser, S. et al. Knee Osteoporosis Diagnosis Based on Deep Learning. Int J Comput Intell Syst 17, 241 (2024). https://doi.org/10.1007/s44196-024-00615-4
  * License: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

* **[Heel Dataset](https://www.kaggle.com/datasets/osamahtaher/heel-dataset)** by Osamah Taher
  * Used for: X-Ray Heel
  * Original Source: Kirkuk General Hospital
  * Citation: Taher, Osamah, and Kasım Özacar. "HeCapsNet: An enhanced capsule network for automated heel disease diagnosis using lateral foot X‐Ray images." International Journal of Imaging Systems and Technology 34.3 (2024): e23084.
  * License: [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

---

## License

Distributed under the MIT License. See `LICENSE.txt` for more information. 

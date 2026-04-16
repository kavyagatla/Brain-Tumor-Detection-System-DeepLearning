Brain Tumor Detection System Using Deep Learning
An automated diagnostic framework designed to bridge the gap between complex deep learning architectures and clinical usability. This system leverages Transfer Learning to classify brain MRI scans into four pathological categories with a peak accuracy of 95.13%.

📌 OverviewManual interpretation of MRI slices by radiologists is a meticulous and time-intensive process prone to human fatigue. This project proposes a Computer-Aided Diagnosis (CAD) system that performs high-fidelity feature extraction to identify subtle pathological patterns often difficult to discern by the naked eye.

Key Features
Multi-Model Ensemble:Comparative analysis across VGG16, ResNet50, and DenseNet121.
High Precision: Achieved 95.13% accuracy and 95.19% precision using the DenseNet121 architecture.\
Clinical Dashboard: A web-based portal for medical professionals to upload scans and view real-time analytics.
Automated Reporting: Generates downloadable PDF diagnostic summaries for patient records.
Data Security: Features a password-protected administrative module for secure data management.

🧬 System Architecture
The system follows a modular 
client-server model:Presentation Layer: Developed using HTML5, CSS3, and Bootstrap for a responsive clinical interface.
Application Layer: Powered by Flask, managing model inference and user authentication.
Data Layer: Utilizes SQLAlchemy and SQLite for secure patient metadata persistence.

📊 Performance Results
Based on empirical testing on a dataset of 3,264 MRI images:
Architecture,Accuracy,Precision,Recall,F1-Score
DenseNet121,95.13%,95.19%,95.13%,95.07%
VGG16,92.40%,92.55%,92.40%,92.26%
ResNet50,83.75%,81.17%,83.75%,82.19%


🔧 Installation & Setup
Clone the Repository
git clone https://github.com/GatlaKavya/Brain-Tumor-Detection-System-AI.git
cd Brain-Tumor-Detection-System-AI
Create a Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies
pip install -r requirements.txt
Run the Application
python main.py

📂 Dataset
The model is trained on the "Brain MRI Images for Brain Tumor Detection" dataset:


Glioma: 926 images 


Meningioma: 937 images 


Pituitary: 901 images 


No Tumor: 500 images

Output Screenshots:
<img width="1435" height="764" alt="image" src="https://github.com/user-attachments/assets/f9f511da-7443-4a88-a2df-9d23a2d46cb0" />
<img width="1357" height="722" alt="image" src="https://github.com/user-attachments/assets/b15d278d-8c08-4bdf-b722-0649c932c060" />
<img width="1300" height="691" alt="image" src="https://github.com/user-attachments/assets/df0eb582-6c13-47e9-85f4-fc5bcc2af3ec" />
<img width="1678" height="693" alt="image" src="https://github.com/user-attachments/assets/f2e89bbd-0ac5-4060-8357-26c4f1fd45a4" />
<img width="1266" height="674" alt="image" src="https://github.com/user-attachments/assets/5df724ff-be28-4a4d-a62a-3614a448bbc0" />






🔮 Future Enhancements
Achieving 100% diagnostic accuracy through hyperparameter optimization.Integrating Explainable AI (XAI) using Grad-CAM to highlight tumor regions.Transitioning to a cloud-based architecture for Telemedicine support.Implementing 3D segmentation for precise tumor volume estimation.

# ♻️ SmartWasteNet

**AI Waste Detection for Circular Economy in Smart Cities**

SmartWasteNet is an AI-powered system designed to classify waste and provide analytics to improve recycling efficiency in smart cities[cite: 1]. Built with Ultralytics YOLOv8 and Streamlit, it detects different types of trash and visualizes actionable sustainability metrics[cite: 1].

## 🚀 Features & Tech Stack

*   **AI-Based Detection:** Utilizes a custom-trained YOLOv8 model for real-time inference[cite: 1].
*   **Analytics Dashboard:** Tracks total waste, recycling efficiency, and estimated CO₂ savings[cite: 1].
*   **Core Technologies:** Python, Streamlit, Ultralytics YOLO, Pandas, and Matplotlib[cite: 1].

## 📂 Project Structure
To run or train this project, ensure your directories are structured as follows:
```text
SmartWasteNet/
├── app.py                  # Main Streamlit application
├── model/
│   └── best.pt             # Trained YOLOv8 weights (Required for app)
├── scripts/
│   ├── download_dataset.py # Fetches the raw dataset from Kaggle
│   ├── convert_dataset.py  # Converts raw dataset to YOLO format
│   └── train.py            # YOLOv8 training script
└── configs/
    └── dataset.yaml        # YOLO dataset configuration
```

## 💻 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/SmartWasteNet.git](https://github.com/yourusername/SmartWasteNet.git)
    cd SmartWasteNet
    ```
2.  **Set up the environment:**
    ```bash
    conda create --name smartwastenet python=3.10 -y
    conda activate smartwastenet
    ```
3.  **Install dependencies:**
    ```bash
    pip install ultralytics streamlit pandas matplotlib pillow torch torchvision
    ```

## 🧠 Dataset & Training

If you wish to train the model from scratch, you will need to manually prepare the dataset and move the resulting weights:

If you wish to train the model from scratch, you do not need to download the dataset manually. Simply run the pipeline:

1.  **Download the Dataset:** Fetches the raw images from Kaggle.
    ```bash
    python scripts/download_dataset.py
    ```
2.  **Format the Dataset:** Converts the downloaded images into the required YOLO bounding box format.
    ```bash
    python scripts/convert_dataset.py
    ```
3.  **Train the Model:**
    Execute the training script to begin the YOLOv8 training process:
    ```bash
    python scripts/train.py
    ```
4.  **Update the Application Weights (Manual Step):**
    Once training is complete, YOLO saves the best weights in a dynamically generated output folder (e.g., `runs/detect/outputs/smartwastenet_training.../weights/best.pt`). 
    *   Navigate to that output folder.
    *   Copy the newly generated `best.pt` file.
    *   Paste it into the `model/` directory in the root of this project, replacing the existing file. The application explicitly loads the model from `model/best.pt`[cite: 1].

## 🏃‍♂️ Usage

**Run the Web Application**
Once your weights are in the correct `model/` folder, launch the interactive Streamlit dashboard:
```bash
streamlit run app.py
```
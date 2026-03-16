# Hair and Scalp Disease Detection System Using CNN

This project uses a Convolutional Neural Network (CNN) to identify different hair and scalp diseases from user-uploaded images. The system includes a Flask web application for real-time predictions and a Jupyter Notebook used to train the model.

## 📁 Project Structure

Hair_Scalp_Detection_System/
│   app.py                      ← Flask web app for predictions  
│   model_training.ipynb        ← Jupyter notebook for training the CNN  
│
├── templates/                  ← HTML templates for Flask  
├── static/                     ← CSS / images / JS files  

> Note: The `hair_disease_best_model.keras` file is included in the final submission ZIP,  
> but NOT uploaded to GitHub due to GitHub’s 100MB file size limit.

---

## ⚙️ Installation & Requirements

Install all required packages:

```bash
pip install -r requirements.txt
```

## 🚀 Running the Flask App
Make sure the model file `hair_disease_best_model.keras` is in the same folder as `app.py`.

Run the Flask app:
```bash
python app.py
```

##  Model Training

The file `model_training.ipynb` contains:

- Dataset preprocessing  
- CNN model building  
- Training & evaluation  
- Saving the model as `.keras`  

---

##  Notes

- GitHub does not allow uploads over **100 MB**, so the model file is only in the submission ZIP.  
- The Flask app loads the model locally and predicts the disease from uploaded images.

---

## Author

Saisha Kalu  
Hair & Scalp Disease Detection — Final Year Project

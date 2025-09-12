# 💰 Discount Analysis System

A **Streamlit-based web application** that analyzes sales, stock, and discount data to generate actionable discount proposals.
The app loads business data from Excel files, applies pre-trained models, and produces a downloadable Excel report with discount recommendations and key metrics.

---

## 🚀 Features

* **Upload multiple Excel files** containing inventory, sales, calendar, and segment data
* **Filter analysis by commercial week range**
* **Segment-based discount proposals**
* Integration with:

  * A **TensorFlow model** (`discount_predictive_model_v2.keras`)
  * A **Gradient Boosting model** (`optimized_gradient_boosting_model.pkl`)
* Preview of the processed data and KPIs:

  * Total items analyzed
  * Items with proposed discounts
  * Average SVA (Stock Value Added)
  * Total residual stock
* **Excel export** of the final report with one click

---

## 📂 Required Files

Before running the app, prepare the following Excel files:

| File Name       | Description                           |
| --------------- | ------------------------------------- |
| `st_item.xlsx`  | Stock Turn items data                 |
| `A.xlsx`        | Main analysis dataset                 |
| `B.xlsx`        | Delta ST data                         |
| `calendar.xlsx` | Calendar mapping with YearWeek values |
| `tracking.xlsx` | Tracking percentages                  |
| `goals.xlsx`    | Function goals                        |
| `segment.xlsx`  | Item segments                         |
| `images.xlsx`   | Image URLs for items                  |
| `sequenza.xlsx` | Discount sequence data                |

> ⚠️ All files must be in `.xlsx` or `.xls` format.

---

## 📦 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/discount-analysis-system.git
cd discount-analysis-system
```

### 2️⃣ Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

> **Typical dependencies:**
>
> * `streamlit`
> * `pandas`
> * `numpy`
> * `tensorflow`
> * `joblib`
> * `openpyxl`
> * `Pillow`

---

## ▶️ Usage

1. Place your trained models in the project directory:

   * `discount_predictive_model_v2.keras`
   * `optimized_gradient_boosting_model.pkl`

2. Run the app:

```bash
streamlit run app.py
```

3. Open your browser at [http://localhost:8501](http://localhost:8501)

4. Upload the required files, configure:

   * **Week range** (e.g., `2024-01` to `2024-52`)
   * **Segments** (e.g., *Fast Fashion*, *Core Collection*, etc.)

5. Click **🚀 Process Analysis** to generate results.

6. Preview KPIs, browse the data, and download the Excel report.

---

## 🧩 Project Structure

```
.
├── app.py                      # Main Streamlit application
├── discount_predictive_model_v2.keras
├── optimized_gradient_boosting_model.pkl
├── requirements.txt
├── README.md
└── sample_data/                # (Optional) Example Excel files
```

---

## ⚠️ Notes

* Ensure your TensorFlow and scikit-learn versions are compatible with the model files.
* Large datasets may increase processing time — progress is displayed in the app.
* For security, **do not upload confidential data** to public instances.

---

## 🛠️ Customization

* Extend the analysis logic in `process_discount_analysis()` to include additional KPIs or custom business rules.
* Update the `categorize_st()` function to adjust ST item clustering logic.
* Modify the Excel styling section to improve report appearance.

---

## 📜 License

This project is released under the [MIT License](LICENSE).

# 📑 Examples

This folder contains an example workflow.

---

## 📂 Data Directory Structure

| **Folder** | **Description** |
|------------|----------------|
| `data/baseline/` | Stores raw **CSV files** before conversion to SQLite. |
| `data/diagnoses/` | " |
| `data/timeseries/measurement_tests_medications/` | " |
| `data/built_/` | Contains the **SQLite database** and **PyTorch dataset** in **processed Parquet files**  after conversion. |

---

## 1️⃣ Building the SQLite database 

Convert your .CSV files into an indexed SQLite database for fast querying

---

## 2️⃣ Building ML-ready Dataloaders

Using the SQLite database, construct a linked Dataset ready for GPU processing.

    📌 creating a dataset → Extracts data from SQLite & generates a deep learning dataset.
    📌 TODO: creating an indexed dataset with outcomes → 

---


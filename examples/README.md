# 📑 Examples

This folder contains an example workflow.

---

## 📂 Data Directory Structure

| **Folder**                                      | **Description**                                                                                            |
|-------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| `data/baseline/`                                | Stores provided raw **CSV files** before conversion to SQLite.                                             |
| `data/diagnoses/`                               | Stores provided raw **CSV files** before conversion to SQLite.                                             |
| `data/timeseries/measurement_tests_medications/` | Stores provided raw **CSV files** before conversion to SQLite.                                             |
| `data/_built/`                                  | Contains the **SQLite database** and **PyTorch dataset** in **processed Parquet files**  after conversion. |

Data provided is entirely demonstrative.

---

## 1. Building the SQLite database 

Convert your .CSV files into an indexed SQLite database for fast querying

---

## 2. Building ML-ready pre-training dataloaders

Using the SQLite database, construct a linked dataset ready for GPU processing.

---

## 3. Building ML-ready supervised dataloaders

Include patient histories up to an index date which can be flexibly specified.

Specify target outcomes.

---

## 4. Modify the FastEHR dataloader for downstream formats

FastEHR provides a standard ragged format for each patient's data. 

In this example we demonstrate how the provided adapter framework allows users to export FastEHR dataloaders to different downstream formats.
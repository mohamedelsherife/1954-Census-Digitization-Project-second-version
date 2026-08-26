# 1954 Census AI Digitization Project

An AI-assisted document digitization project for extracting structured information from historical **1954 census documents**.

The project aims to transform scanned census pages into structured, machine-readable data by combining:

* Image preprocessing
* Document parsing and segmentation
* OCR for printed text
* HTR for handwritten text
* Post-processing
* Structured data extraction
* Human review for difficult or ambiguous fields

---

## 1. Project Overview

The source documents are historical census pages from **1954**.

Each page contains a large table with data for up to **three families**, distributed across approximately **22 columns**.

The documents contain a mixture of:

* Printed Arabic text
* Official headers and column titles
* Handwritten Arabic data
* Handwritten notes and annotations
* Different ink colors
* Table lines and borders

Because the documents are old and contain handwritten information, a single OCR system is not sufficient.

The project therefore uses two complementary text-recognition paths:

```text
Printed Text  → OCR
Handwriting   → HTR
```

The results are later combined and transformed into structured data.

---

# 2. Project Goals

The main goals of this project are:

1. Digitize historical 1954 census documents.
2. Improve the quality of scanned images using image preprocessing.
3. Identify the structure of the census document.
4. Segment the document into meaningful regions and cells.
5. Extract printed text using OCR.
6. Extract handwritten Arabic text using HTR.
7. Handle difficult or ambiguous fields through post-processing.
8. Convert extracted information into structured data.
9. Preserve the relationship between fields, rows, and families.
10. Document the complete workflow and publish the project through GitHub.

---

# 3. Document Structure

Each census page can be divided into several major components.

```text
Page
│
├── Header
│   ├── Page information
│   └── Official information
│
├── Printed Elements
│   └── Column titles
│
└── Census Table
    ├── Family 1
    ├── Family 2
    └── Family 3
```

The table contains multiple columns representing different types of census information.

Some information is written inside the table, while additional notes may appear inside or around the table.

---

# 4. Main Challenges

The historical documents introduce several challenges that must be considered during processing.

### 4.1 Image Quality

The original scans contain noise and degradation that can reduce recognition accuracy.

### 4.2 Rotation and Skew

Some pages are not perfectly aligned.

This can affect:

* Table detection
* Cell detection
* OCR
* HTR

Therefore, deskewing is considered an important preprocessing step.

### 4.3 Handwriting Legibility

Some handwritten information is difficult to read even for a human.

This makes handwritten text recognition particularly challenging.

### 4.4 Notes and Annotations

Handwritten notes can appear inside empty columns or between fields.

This creates a risk of assigning a note to the wrong field.

### 4.5 The `///` Repeat Mark

Some cells contain a repeat mark such as:

```text
///
```

This mark indicates that the value is the same as the value from the previous row.

It should therefore be handled as a special case during post-processing rather than simply being treated as normal text.

### 4.6 Ambiguous Column Labels

The age-related column may have different labels in different documents, such as an age-related label or a date-of-birth label.

A field-mapping strategy may therefore be required.

### 4.7 Complex Fields

Some fields are more difficult to automatically recognize, including:

* Occupation
* Marital status
* Educational status
* Total family information

These fields may require additional post-processing or human review.

---

# 5. Overall Workflow

The complete project pipeline is:

```text
Original PDF
     │
     ▼
PDF → JPG
     │
     ▼
Image Preprocessing
     │
     ├── Grayscale
     ├── Thresholding
     ├── Noise Removal
     ├── Cropping
     └── Deskewing
     │
     ▼
Document Parsing / Segmentation
     │
     ├── Header
     ├── Table
     ├── Rows
     ├── Columns
     └── Cells
     │
     ├───────────────────────┐
     ▼                       ▼
Print
ed Text             Handwritten Text
     │                       │
     ▼                       ▼
    OCR                     HTR
     │                       │
     └───────────┬───────────┘
                 ▼
          Post-processing
                 │
                 ▼
        Structured Data
                 │
                 ▼
          Final Output
```

The document parsing stage separates the page into meaningful regions before the OCR and HTR stages.

---

# 6. Image Preprocessing

Image preprocessing is used to improve the quality and consistency of the historical scans before recognition.

The preprocessing pipeline includes operations such as:

### Grayscale

Converts the image into a grayscale representation.

```text
Original Image
      ↓
Grayscale Image
```

### Adaptive Thresholding

Thresholding can improve the visibility of text and table structures.

```text
Grayscale
    ↓
Adaptive Threshold
    ↓
Binary / Enhanced Image
```

### Noise Removal

Removes unwanted image artifacts while attempting to preserve important text and table structures.

### Cropping

Unnecessary borders, dark margins, and irrelevant parts of the document can be removed.

Cropping is also used to isolate specific regions such as the header or table.

### Deskewing

Some documents contain rotation or skew.

A deskewing stage can be used to align the document before further processing.

---

# 7. Document Parsing and Segmentation

Document Parsing is the stage responsible for understanding the **layout and structure** of the document.

It does not primarily read the text.

Instead, it answers questions such as:

* Where is the header?
* Where does the table begin?
* Where does the table end?
* Where are the rows?
* Where are the columns?
* Where are the individual cells?
* Which image region should be sent to OCR?
* Which region should be sent to HTR?

In simple terms:

```text
Document Parsing → Where is the information?

OCR              → What printed text is there?

HTR              → What handwritten text is there?
```

---

## 7.1 Bounding Boxes

Bounding boxes are rectangular regions used to identify specific areas of the document.

For example:

```text
┌───────────────────────┐
│                       │
│       Cell Region     │
│                       │
└───────────────────────┘
```

A bounding box can be represented using coordinates such as:

```text
x1, y1, x2, y2
```

These coordinates can then be used to crop the corresponding region.

---

## 7.2 Grid Detection

The census document contains a table structure.

Grid detection attempts to identify:

* Horizontal lines
* Vertical lines
* Table boundaries
* Cell boundaries

Conceptually:

```text
│       │       │       │
│ Cell  │ Cell  │ Cell  │
├───────┼───────┼───────┤
│ Cell  │ Cell  │ Cell  │
├───────┼───────┼───────┤
│ Cell  │ Cell  │ Cell  │
```

The detected grid can be used to determine the locations of individual cells.

---

## 7.3 Part-Line Detection

Historical documents do not always contain perfect continuous table lines.

Some lines may be:

* Broken
* Faint
* Distorted
* Covered by handwriting
* Missing in some areas

Part-line detection can therefore help identify partial horizontal or vertical lines and use them as additional information when reconstructing the table structure.

---

## 7.4 Relationship Between the Methods

Bounding boxes, grid detection, and part-line detection are approaches that can support the overall Document Parsing process.

They are not necessarily three independent stages.

A possible workflow is:

```text
Detect Table Grid
       ↓
Detect Available Lines
       ↓
Identify Cell Boundaries
       ↓
Generate Bounding Boxes
       ↓
Crop Cells
       ↓
Send Cells to OCR / HTR
```

The project evaluates these approaches according to how well they work with the historical census pages.

---

# 8. OCR — Printed Text Recognition

OCR stands for:

**Optical Character Recognition**

OCR is used to recognize printed text.

In this project, OCR is primarily intended for:

* Printed headers
* Official information
* Printed column titles
* Other machine-readable printed regions

The general process is:

```text
Printed Crop
     ↓
Preprocessing
     ↓
OCR
     ↓
Extracted Text
```

The OCR output can then be cleaned and mapped to the corresponding field.

---

# 9. HTR — Handwritten Text Recognition

HTR stands for:

**Handwritten Text Recognition**

HTR is used to recognize handwritten information in the census forms.

This is especially important because many of the actual census values are handwritten in Arabic.

The general process is:

```text
Handwritten Cell
       ↓
Image Preparation
       ↓
HTR Model
       ↓
Recognized Text
       ↓
Post-processing
```

HTR is different from OCR because the input contains handwritten rather than printed text.

---

# 10. OCR vs HTR

| Task             | Technology       | Main Purpose                          |
| ---------------- | ---------------- | ------------------------------------- |
| Printed text     | OCR              | Recognize printed information         |
| Handwritten text | HTR              | Recognize handwritten information     |
| Layout           | Document Parsing | Identify where information is located |
| Image quality    | OpenCV           | Improve input images                  |
| Final correction | Post-processing  | Clean and structure extracted results |

The project therefore does not rely on a single recognition technology.

---

# 11. Post-Processing

Recognition results should not immediately be considered final data.

Post-processing is responsible for cleaning, correcting, and structuring the extracted results.

Possible operations include:

### Handling `///`

If:

```text
Row 1 → محمد
Row 2 → ///
```

the system should interpret:

```text
Row 2 → محمد
```

rather than storing `///` as the actual value.

### Field Mapping

Different labels may refer to the same logical field.

A dictionary or lexicon can help map variants to standardized field names.

Example:

```text
Different label
      ↓
Standard field name
      ↓
age
```

### Human-in-the-Loop Review

Some information may be too ambiguous for fully automatic recognition.

For difficult fields, the system can flag the result for human verification.

```text
HTR/OCR Result
      ↓
Confidence / Validation
      ↓
 ┌────┴─────┐
 │          │
Reliable   Uncertain
 │          │
 ▼          ▼
Accept    Human Review
```

---

# 12. Structured Data

The final objective is not simply to produce a block of text.

The extracted information should be organized into structured records.

Conceptually:

```text
Page
│
├── Family 1
│   ├── Field 1
│   ├── Field 2
│   ├── Field 3
│   └── ...
│
├── Family 2
│   ├── Field 1
│   ├── Field 2
│   ├── Field 3
│   └── ...
│
└── Family 3
    ├── Field 1
    ├── Field 2
    ├── Field 3
    └── ...
```

Possible final formats include:

* CSV
* JSON
* Database records

The exact final output format can be selected during the implementation stage.

---

# 13. Project File Structure

The planned project structure is:

```text
1954-Census-AI-Digitization-Project/
│
├── data/
│   ├── raw/
│   ├── cropped/
│   ├── cells/
│   └── processed/
│
├── output/
│   ├── crops/
│   ├── ocr/
│   └── htr/
│
├── scripts/
│   ├── preprocessing.py
│   ├── document_parsing.py
│   ├── ocr.py
│   └── htr.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

# 14. Directory Description

## `data/raw/`

Contains the original input documents and images.

These files should remain unchanged so that the original source is preserved.

---

## `data/cropped/`

Contains images produced after larger-scale cropping operations.

Examples may include:

```text
header
table
page regions
```

---

## `data/cells/`

Contains individual table cells or smaller segmented regions.

These images can be used as inputs for OCR or HTR.

---

## `data/processed/`

Contains images after preprocessing operations such as:

* Grayscale
* Thresholding
* Noise removal
* Other image enhancement operations

---

## `output/crops/`

Contains crops generated specifically for recognition.

For example:

```text
header crops
printed regions
handwritten regions
```

---

## `output/ocr/`

Contains OCR results.

Possible output:

```text
.txt
.json
.csv
```

depending on the final implementation.

---

## `output/htr/`

Contains HTR results and/or outputs generated by the handwritten text recognition pipeline.

---

## `scripts/preprocessing.py`

Responsible for image preprocessing operations.

Possible responsibilities:

* Loading images
* Grayscale conversion
* Thresholding
* Noise removal
* Cropping
* Deskewing

---

## `scripts/document_parsing.py`

Responsible for analyzing the document layout.

Possible responsibilities:

* Detecting table regions
* Detecting lines
* Detecting grid structures
* Generating bounding boxes
* Identifying rows and columns
* Generating cell crops

---

## `scripts/ocr.py`

Responsible for the printed text recognition pipeline.

```text
Printed Image
      ↓
OCR
      ↓
Text
```

---

## `scripts/htr.py`

Responsible for the handwritten text recognition pipeline.

```text
Handwritten Image
       ↓
HTR Model
       ↓
Text
```

---

## `main.py`

Acts as the main entry point for running the complete pipeline.

A future implementation may connect the different stages:

```text
Preprocessing
      ↓
Document Parsing
      ↓
OCR / HTR
      ↓
Post-processing
      ↓
Structured Output
```

---

## `requirements.txt`

Contains the Python dependencies required by the project.

The exact dependencies should reflect the libraries actually used in the implementation.

---

# 15. Example Pipeline

For a single census page:

```text
1954-P000001.pdf
        │
        ▼
Convert PDF → JPG
        │
        ▼
Preprocess Image
        │
        ├── Grayscale
        ├── Threshold
        ├── Noise Removal
        └── Crop
        │
        ▼
Document Parsing
        │
        ├── Header
        └── Table
             │
             ├── Rows
             └── Cells
                  │
            ┌─────┴─────┐
            ▼           ▼
          OCR          HTR
            │           │
            └─────┬─────┘
                  ▼
            Post-processing
                  │
                  ▼
           Structured Data
```

---

# 16. Current Project Progress

The project is being developed incrementally.

### Completed

* [x] Manual image inspection
* [x] Documentation of source-image problems
* [x] PDF to JPG conversion
* [x] Initial image preprocessing
* [x] Image cropping
* [x] Initial document segmentation
* [x] Experiments with document parsing
* [x] Experiments with bounding boxes
* [x] Experiments with grid detection
* [x] Experiments with part-line detection

### In Progress

* [ ] OCR pipeline
* [ ] HTR pipeline
* [ ] Selecting/finalizing the most suitable document parsing approach
* [ ] Improving recognition accuracy
* [ ] Post-processing

### Planned

* [ ] Structured data generation
* [ ] Validation and human review
* [ ] Final result storage
* [ ] Complete documentation
* [ ] GitHub publishing

---

# 17. Design Philosophy

The project follows a modular approach.

Instead of attempting to recognize the entire page in one step, the document is progressively transformed:

```text
Complex Historical Document
          ↓
Cleaned Image
          ↓
Document Structure
          ↓
Individual Regions
          ↓
Individual Cells
          ↓
OCR / HTR
          ↓
Clean Text
          ↓
Structured Data
```

This approach makes it easier to:

* Debug individual stages
* Improve preprocessing independently
* Test OCR and HTR separately
* Replace recognition models
* Identify errors
* Review difficult fields
* Maintain the project

---

# 18. Important Recognition Strategy

The project treats printed and handwritten information differently.

```text
                Census Page
                     │
              Document Parsing
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
    Printed Regions       Handwritten Regions
          │                     │
          ▼                     ▼
         OCR                   HTR
          │                     │
          └──────────┬──────────┘
                     ▼
              Post-processing
                     │
                     ▼
             Structured Data
```

This separation is important because the source document contains both printed and handwritten information.

---

# 19. Data Quality and Validation

Recognition accuracy is affected by the quality of the original historical documents.

The project therefore considers:

* Image quality
* Noise
* Skew
* Handwriting legibility
* Table structure
* Ambiguous fields
* Notes and annotations
* Repeated-value marks
* OCR/HTR recognition errors

Not every field should be assumed to be correctly recognized automatically.

For difficult fields, human validation can be incorporated into the workflow.

---

# 20. Future Improvements

Potential improvements include:

1. Better deskewing.
2. Improved table-line detection.
3. More robust cell segmentation.
4. Better handling of broken table lines.
5. Improved Arabic OCR preprocessing.
6. HTR model evaluation.
7. Confidence-based human review.
8. Automatic handling of repeated values.
9. Standardized field mapping.
10. Improved structured-data validation.

---

# 21. Conclusion

The **1954 Census AI Digitization Project** combines traditional image processing with OCR and HTR to convert historical census documents into structured digital information.

The project follows a modular pipeline:

```text
Image
  ↓
Preprocessing
  ↓
Document Parsing
  ↓
Segmentation
  ↓
OCR + HTR
  ↓
Post-processing
  ↓
Structured Data
```

The main challenge is not simply recognizing text. The system must first understand the structure of the historical document, separate printed and handwritten information, correctly identify table cells, and then transform recognition results into reliable structured records.

Because the source documents contain noise, skew, handwritten annotations, ambiguous fields, and difficult handwriting, the project also considers validation and human review as part of the overall digitization process.

---

## Project Team

* Issa Ramy Alsherife
* Othman Alfatouri
* Mohammed Ramy Alsherife

---

## Project Status

**1954 Census AI Digitization Project — In Development**

The current development stage focuses on:

**Document Parsing → OCR → HTR → Post-processing → Structured Data**

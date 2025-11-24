# Biomaterial-Selection-and-Bioinformatics-Tool

## NOTE-- This README is incomplete as of now, basic operational instructions only, complete details coming very soon!

A modern Streamlit-based biomaterials intelligence platform combining:

* **Local database search** (details coming soon...)
* **Materials Project API integration** (real-time crystal structure, elasticity, magnetism, bonding, oxidation, and thermodynamic data) details coming soon
* **3D structure visualization** using **py3Dmol**
    - Visualisation not yet implemented for 3D Structures of elements/materials, more details coming soon...
* **Machine-learning pipelines** for structure prediction (details coming ...)

This project is part of a larger effort to build an intelligent decision-support system for materials science and biomaterials engineering.

---

Platform is live on Streamlit!

[Visit The app on Streamlit](https://biomat-test-inuxwupzxu4apppplsy93ya2.streamlit.app/)

---

# Datasets and citation

...

---

## **Features**

### 🔍 **Local Database Search**

* Query materials stored in your local CSV or database.
* Fast and cached lookup.
* Returns full material metadata.

---

### 🌐 **Materials Project Integration**

* Live lookup of:

  * Structure summaries
  * Elasticity data (when available)
  * Magnetism
  * Bonding information
  * Oxidation states
  * Surface properties
  * Thermodynamics (when available)
* Handles missing MP data gracefully.

---

### 🧪 **3D Structure Viewer** (not yet implemented...)

* Interactive **py3Dmol** molecular visualizer. (details coming soon...)
* Supports rotation, zoom, and style changes.
* Automatically loads valid MP structures.

---

### 🧮 **ML Prediction Pipelines**

* Details coming soon ...

---

### 🖥️ **Modern Streamlit Dashboard**

* Clean UI with custom CSS.
* Uses `st.session_state` for stable navigation.
* Mobile-aware responsive design.

---

## **Project Structure** (full structure explained soon...)

```
project/
│
├── streamlit_app.py           # Main multi-page Streamlit entrypoint
├── visualization.py           # MP API data visualization tab
├── mp_integration.py          # API integration helper functions
├── data/                      # Local datasets
├── models/                    # Trained ML models
├── utils/
│   ├── ml_pipelines/
│   │   └── crystalline_structure_predictor.py
│   └── ...
└── README.md
```

---

## **Installation**

### **1. Clone the repo**

```bash
git clone <https://github.com/Hdave00/Biomaterial-Selection-and-Bioinformatics-Tool>
cd project
```

### **2. Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### **3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## **Environment Variables**

Create a `.env` file in the project root:

```
MP_API_KEY=your_materials_project_api_key_here
```

This key is loaded in `mp_integration.py` using **python-dotenv**.

---

## **Running the App**

```bash
streamlit run streamlit_app.py
```

---

## **Usage Overview**

### **Local Search**

1. Navigate to **Local Search** tab.
2. Type a material name or ID.
3. Results appear instantly by caching.

### **Materials Project Lookup**

1. Select an element → structure → MP Entry ID.
2. View:

   * Material summary
   * Bonding + magnetism
   * Elasticity (when available)


---
### **ML Prediction**

1. Navigate to **Prediction** tab. (more details coming soon...)

---

## **Future Roadmap (Coming Soon...)**

* ML-enhanced materials recommendation engine
* Implant-specific constraints (shear force, cytotoxicity, budget, lifespan)
* Automated structure report PDF generator
* WASM-powered C modules for simulation
* User profiles & saved searches

---

## **Troubleshooting**

### **No elasticity data**

Some MP entries simply do not contain elasticity measurements.
The UI will show:

> *“No elasticity data available for this entry.”*

### **Sub-types not appearing**

This happens if MP returns **no structures** for your element.
Try selecting another element or refresh the tab.

---

## **License**

MIT


---

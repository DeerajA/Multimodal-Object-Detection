# 🧠 Multi-Agent Visual Question Answering System (YOLO + LLM Planner)

## 📌 Project Overview

This project is a **multi-agent visual question answering (VQA) system** that combines:

* **YOLO object detection (Ultralytics)** for image understanding
* **A large language model (LLM)** for reasoning and planning
* **A modular agent architecture** where each agent performs a specific vision-related task

The system takes:

* a **natural language question**
* an **image path**

…and automatically determines **which vision agents are required** to answer that question.

---

## 🎯 End Goal of the Project

The ultimate goal of this project is to:

> **Automatically answer image-based questions by dynamically selecting and orchestrating specialized vision agents using an LLM planner.**

Instead of running all computer vision functions blindly, the system:

1. **Analyzes the question**
2. **Selects only the necessary agents**
3. **Executes them to extract the required information**

This architecture is scalable, interpretable, and efficient.

---

## 🧩 High-Level Architecture

```
User Question + Image
        │
        ▼
🧠 Planner Agent (LLM)
        │
        ▼
Select Required Agents
        │
        ▼
🛠️ Vision Agents (YOLO-based)
        │
        ▼
Structured Outputs (counts, locations, image size, visualizations)
```

---

## 📁 Project Structure

```
project/
│
├── DetectionAgent.py        # Object counting
├── LocationAgent.py         # Bounding box locations
├── SizeAgent.py             # Image dimensions
├── VisualizationAgent.py    # YOLO visual output
│
├── ParentAgent.py           # Planner + agent selection logic
├── state.py                 # Shared state definition
│
├── main.py                  # Entry point / example usage
├── images/
│   └── image2.png
│
└── README.md
```

---

## 🤖 Vision Agents (YOLO-Based)

All agents use the **Ultralytics YOLO model (`yolo11n.pt`)**.

---

### 1️⃣ DetectionAgent – Object Counting

**Purpose:**
Counts how many instances of each object class appear in the image.

**File Logic:**

```python
def getCount(img_path) -> dict
```

**Output Example:**

```json
{
  "person": 3,
  "car": 1,
  "dog": 2
}
```

**Use Case:**

* "How many people are in the image?"
* "Count the objects in this picture."

---

### 2️⃣ LocationAgent – Object Locations

**Purpose:**
Returns bounding box coordinates for each detected object.

**File Logic:**

```python
def getLocations(img_path) -> List[str]
```

**Output Example:**

```
Name: person, Location: (x1, y1, x2, y2)
```

**Use Case:**

* "Where are the objects located?"
* "Give bounding box coordinates."

---

### 3️⃣ SizeAgent – Image Dimensions

**Purpose:**
Returns the width and height of the input image.

**File Logic:**

```python
def getSize(img_path) -> (width, height)
```

**Use Case:**

* "What is the size of the image?"
* "How large is the image resolution?"

---

### 4️⃣ VisualizationAgent – Detection Visualization

**Purpose:**
Displays the YOLO-detected image with bounding boxes.

**File Logic:**

```python
def show_plot(img_path)
```

**Output:**

* Matplotlib window showing detected objects.

**Use Case:**

* Debugging
* Visual confirmation of detections

---

## 🧠 Planner Agent (LLM-Based)

### 🔍 What It Does

The planner uses an **LLM (DeepSeek-R1 via NVIDIA API)** to:

* Read the **user question**
* Decide **which agents are required**
* Return a structured list of agent names

---

### 📄 Planner Prompt

The planner is instructed using a structured prompt with examples and strict output formatting enforced via **Pydantic**.

**Available Agents:**

* `DetectionAgent`
* `VisualizationAgent`
* `LocationAgent`
* `SizeAgent`

---

### 🧾 Planner Output Schema

```python
class template(BaseModel):
    requiredAgents: List[str]
```

**Example Output:**

```json
{
  "requiredAgents": ["DetectionAgent"]
}
```

---

## 📦 State Management

### `state.py`

Defines the shared state passed across the system:

```python
class state:
    question: str
    image_path: str
    requiredAgents: List[str]
```

This allows:

* Clean separation of concerns
* Easy extension for future agents (e.g., depth, segmentation)

---

## ▶️ Example Execution Flow

### `main.py`

```python
state["question"] = "How many people are in the image?"
state["image_path"] = "images/image2.png"

result = get_agents(state)
print(result)
```

### What Happens Internally

1. The **planner agent** reads the question
2. It determines that `DetectionAgent` is required
3. The selected agents can then be executed to answer the question

---

## 🚀 Key Strengths of This Design

✅ Modular and extensible agent-based architecture
✅ Efficient (only required agents are run)
✅ Interpretable decision-making via LLM planning
✅ Combines symbolic reasoning with deep learning
✅ Ideal foundation for advanced VQA systems

---

## 🔮 Future Extensions

* Add a **Response Agent** to generate natural language answers
* Integrate **agent execution loop** after planning
* Support **multiple images**
* Add **segmentation or pose estimation agents**
* Persist results using Neptune or other experiment trackers

---

## 🧠 Summary

This project demonstrates a **modern AI system design** that fuses:

* **Computer vision**
* **LLM-based reasoning**
* **Multi-agent orchestration**

It is not just detecting objects—it is **thinking about how to answer questions intelligently**.

If you want, I can next:

* Add agent execution logic
* Convert this into a full VQA pipeline
* Or refactor it into a production-ready framework

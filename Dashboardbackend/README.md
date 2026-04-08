# Backend - Dashboard Project

This directory contains the backend portion of the Dashboard Project, built with FastAPI.

## 🚀 Features
- **Role-Based Access**: Role-based access control and special endpoints.
- **Real-time Streaming**: WebSockets for pushing real-time analytics to the frontend.
- **Modern Tech Stack**: FastAPI (Python), Motor (Async MongoDB), Pydantic.

## 💻 Local Development

### Prerequisites
- Python (v3.11+)
- MongoDB

### Setup & Run
1. **Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\\Scripts\\activate
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   Create `.env`:
   ```env
   MONGO_URI=mongodb://localhost:27017/devdb?replicaSet=rs0
   DB_NAME=devdb,air,road,ocean and courier
   SECRET_KEY=your_secret_key
   ALLOWED_ORIGINS=http://localhost:5173
   ```

3. **Start the server**:
   ```bash
   uvicorn main:app --port 8000 --reload
   ```

## 📦 Deployment
You can deploy the backend using the provided `Dockerfile`. For details on a full stack deployment including frontend and database, refer to the docker-compose setup in the root repository.

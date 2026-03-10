# How to Run the Project

## Backend

# The following code works in python 3.13.

1. cd backend  
2. python -m venv venv  
3. venv\Scripts\activate  
4. pip install -r requirements.txt  
5. Go to backend\core\config.py. Here you need to change MySQL user and password for the backend to work.
6. python -m uvicorn app:app --host 127.0.0.1 --port 8000  
7. go to http://127.0.0.1:8000/docs   
8. go to post/infer/review  
9. try it out  

## Frontend

1. cd frontend  
2. npm install  
3. npm run dev  
4. try it out  

## Graph Views

- Single review explanation graph:
  1. Start backend and frontend.
  2. Submit a review in the `Single Review` form.
  3. Open the `Review Explanation` graph mode on the dashboard.

- Batch / corpus aspect graph:
  1. Upload a CSV in the `Batch CSV` form.
  2. Switch to `Corpus Analytics` graph mode.
  3. Use the graph filter bar to change `domain`, `product_id`, date range, and minimum edge weight.

## Graph API

- `GET /graph/review/{review_id}` returns the single-review explanation graph.
- `GET /graph/aspects` returns the batch co-occurrence graph.

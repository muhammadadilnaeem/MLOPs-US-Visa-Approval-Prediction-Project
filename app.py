
# Import necessary modules
import os
from typing import Optional
from fastapi import FastAPI, Request
from uvicorn import run as app_run
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, RedirectResponse

# Import project-specific modules and constants
from us_visa_project.constants import APP_HOST, APP_PORT
from us_visa_project.pipline.prediction_pipeline import USvisaData, USvisaClassifier
from us_visa_project.pipline.training_pipeline import TrainPipeline

# Initialize the FastAPI app
app = FastAPI()

# Mount static files (like CSS, JavaScript, images) for the app
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates for rendering HTML templates
templates = Jinja2Templates(directory='templates')

# Define CORS (Cross-Origin Resource Sharing) settings to allow any origin to access the API
origins = ["*"]

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,           # Allow requests from any origin
    allow_credentials=True,           # Allow sending cookies with requests
    allow_methods=["*"],              # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],              # Allow all headers in requests
)

# Define a class to handle form data from requests
class DataForm:
    """
    DataForm class to handle and store data from HTML form submissions.
    """

    def __init__(self, request: Request):
        """
        Initialize DataForm with optional fields for each data point.
        :param request: The request object containing form data.
        """
        self.request: Request = request
        self.continent: Optional[str] = None
        self.education_of_employee: Optional[str] = None
        self.has_job_experience: Optional[str] = None
        self.requires_job_training: Optional[str] = None
        self.no_of_employees: Optional[str] = None
        self.company_age: Optional[str] = None
        self.region_of_employment: Optional[str] = None
        self.prevailing_wage: Optional[str] = None
        self.unit_of_wage: Optional[str] = None
        self.full_time_position: Optional[str] = None

    async def get_usvisa_data(self):
        """
        Populate DataForm fields with data from the submitted form.
        """
        form = await self.request.form()
        self.continent = form.get("continent")
        self.education_of_employee = form.get("education_of_employee")
        self.has_job_experience = form.get("has_job_experience")
        self.requires_job_training = form.get("requires_job_training")
        self.no_of_employees = form.get("no_of_employees")
        self.company_age = form.get("company_age")
        self.region_of_employment = form.get("region_of_employment")
        self.prevailing_wage = form.get("prevailing_wage")
        self.unit_of_wage = form.get("unit_of_wage")
        self.full_time_position = form.get("full_time_position")

# Define the root route ("/") to render the main HTML page
@app.get("/", tags=["authentication"])
async def index(request: Request):
    """
    Render the main HTML page (usvisa.html) for data input.
    :param request: The request object.
    :return: Rendered HTML page with context.
    """
    return templates.TemplateResponse(
        "index.html", {"request": request, "context": "Rendering"}
    )

# Define the "/train" route to trigger the model training pipeline
@app.get("/train")
async def trainRouteClient():
    """
    Run the training pipeline and return a success message.
    :return: Response indicating training success or failure.
    """
    try:
        train_pipeline = TrainPipeline()   # Initialize the training pipeline
        train_pipeline.run_pipeline()      # Run the training pipeline

        return Response("Training successful !!")   # Send success response

    except Exception as e:
        return Response(f"Error Occurred! {e}")      # Send error response in case of failure

# Define the POST route ("/") to handle prediction requests
@app.post("/")
async def predictRouteClient(request: Request):
    """
    Handle form submission, perform prediction, and return results.
    :param request: The request object with form data.
    :return: Rendered HTML page with prediction result.
    """
    try:
        # Parse form data and create a data object for prediction
        form = DataForm(request)
        await form.get_usvisa_data()       # Retrieve data from the form

        # Create an instance of USvisaData with parsed form data
        usvisa_data = USvisaData(
            continent=form.continent,
            education_of_employee=form.education_of_employee,
            has_job_experience=form.has_job_experience,
            requires_job_training=form.requires_job_training,
            no_of_employees=form.no_of_employees,
            company_age=form.company_age,
            region_of_employment=form.region_of_employment,
            prevailing_wage=form.prevailing_wage,
            unit_of_wage=form.unit_of_wage,
            full_time_position=form.full_time_position,
        )

        # Convert data to DataFrame format for model prediction
        usvisa_df = usvisa_data.get_usvisa_input_data_frame()

        # Instantiate the model classifier and perform prediction
        model_predictor = USvisaClassifier()
        value = model_predictor.predict(dataframe=usvisa_df)[0]

        # Determine visa approval status based on prediction
        status = "Visa-approved" if value == 1 else "Visa Not-Approved"

        # Render the HTML template with prediction result
        return templates.TemplateResponse(
            "usvisa.html",
            {"request": request, "context": status},
        )

    except Exception as e:
        # Return error message in case of failure
        return {"status": False, "error": f"{e}"}

# Run the application with specified host and port
if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)


"""
Pydantic Schemas for FastAPI Request/Response Validation.
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field


class HeartDiseaseInput(BaseModel):
    """
    Schema for individual heart disease prediction request payload.
    """

    age: float = Field(..., description="Age in years", json_schema_extra={"example": 57.0})
    sex: float = Field(..., description="Sex (1 = male; 0 = female)", json_schema_extra={"example": 1.0})
    cp: float = Field(
        ...,
        description="Chest pain type (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic)",
        json_schema_extra={"example": 4.0},
    )
    trestbps: float = Field(
        ..., description="Resting blood pressure in mm Hg on admission", json_schema_extra={"example": 140.0}
    )
    chol: float = Field(..., description="Serum cholesterol in mg/dl", json_schema_extra={"example": 241.0})
    fbs: float = Field(
        ..., description="Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)", json_schema_extra={"example": 0.0}
    )
    restecg: float = Field(
        ...,
        description="Resting electrocardiographic results (0: normal, 1: ST-T wave abnormality, 2: LV hypertrophy)",
        json_schema_extra={"example": 1.0},
    )
    thalach: float = Field(..., description="Maximum heart rate achieved", json_schema_extra={"example": 123.0})
    exang: float = Field(
        ..., description="Exercise induced angina (1 = yes; 0 = no)", json_schema_extra={"example": 1.0}
    )
    oldpeak: float = Field(
        ..., description="ST depression induced by exercise relative to rest", json_schema_extra={"example": 1.5}
    )
    slope: float = Field(
        ...,
        description="Slope of peak exercise ST segment (1: upsloping, 2: flat, 3: downsloping)",
        json_schema_extra={"example": 2.0},
    )
    ca: float = Field(
        ..., description="Number of major vessels (0-3) colored by fluoroscopy", json_schema_extra={"example": 0.0}
    )
    thal: float = Field(
        ..., description="Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)", json_schema_extra={"example": 7.0}
    )

    def to_dict(self) -> Dict[str, float]:
        """Converts input object to dictionary matching dataframe columns."""
        return {
            "age": self.age,
            "sex": self.sex,
            "cp": self.cp,
            "trestbps": self.trestbps,
            "chol": self.chol,
            "fbs": self.fbs,
            "restecg": self.restecg,
            "thalach": self.thalach,
            "exang": self.exang,
            "oldpeak": self.oldpeak,
            "slope": self.slope,
            "ca": self.ca,
            "thal": self.thal,
        }


class PredictionOutput(BaseModel):
    """
    Schema for heart disease prediction response payload.
    """

    prediction: int = Field(..., description="Binary prediction: 1 = Disease Present, 0 = Absent")
    heart_disease_risk: str = Field(..., description="Human readable risk assessment")
    probability: float = Field(..., description="Predicted probability of heart disease (0.0 to 1.0)")
    model_version: str = Field(..., description="MLflow registered model version or stage")


class HealthResponse(BaseModel):
    """
    Schema for readiness/health check response.
    """

    status: str = Field(..., description="Application health status")
    version: str = Field(..., description="API Version")
    model_loaded: bool = Field(..., description="Whether model is loaded and ready for inference")

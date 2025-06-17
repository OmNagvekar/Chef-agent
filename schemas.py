from pydantic import BaseModel,Field
from typing import Dict,List,Any,Optional,Literal
import json

class Recipe(BaseModel):
    """
    Represents a recipe with its details.
    """
    title: str = Field(default=None, description="The title of the recipe")
    description: Optional[str] = Field(default=None, description="A brief description of the recipe")
    veg_nonveg: Literal["veg", "nonveg"] = Field(default=None, description="Indicates if the recipe is vegetarian or non-vegetarian")
    ingredients: List[str] = Field(default_factory=list, description="List of ingredients required for the recipe")
    instructions: List[str] = Field(default_factory=list, description="Step-by-step instructions for preparing the recipe")
    image_url: Optional[str] = Field(default=None, description="URL of the recipe image")
    video_url: Optional[str] = Field(default=None, description="URL of a video demonstrating the recipe")
    prep_time: Optional[str] = Field(default=None, description="Preparation time for the recipe")
    cook_time: Optional[str] = Field(default=None, description="Cooking time for the recipe")
    servings: Optional[int] = Field(default=None, description="Number of servings the recipe yields")
    cuisine: Optional[str] = Field(default=None, description="Type of cuisine (e.g., Italian, Mexican)")
    category: Optional[str] = Field(default=None, description="Category of the recipe (e.g., dessert, main course)")
    nutrition: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Nutritional information of the recipe, such as calories, fat, protein, etc."
    ) 
    # e.g., {'calories': '200 kcal', 'fat': '10g'}
    
    def to_json_string(self):
        """Converts the Data object to a JSON string."""
        return json.dumps(self.model_dump(mode='json'), indent=4)
    
class Profile(BaseModel):
    """
    Represents a user profile with basic information.
    """
    name: str = Field(default=None, description="Name of the user")
    recipes: List[Recipe] = Field(
        default_factory=list, 
        description="List of recipes saved by the user"
    )
    preferences: Dict[str, Any] = Field(
        default_factory=dict, 
        description="User preferences such as dietary restrictions, favorite cuisines, etc."
    )
    
    def to_json_string(self):
        """Converts the Profile object to a JSON string."""
        return json.dumps(self.model_dump(mode='json'), indent=4)
    
class UpdateGraphDecision(BaseModel):
    """
    Represents a decision to update the graph with new data.
    """
    should_update: bool = Field(
        default=False, 
        description="Set to True if the graph needs to be updated or new information added"
    )
    tool_choice:Literal["graph_query","ingest_url_to_graph","both","none"] = Field(
        default="none", 
        description="The tool to use for updating the graph. Options are 'graph_query', 'ingest_url_to_graph', or 'both'."
    )
    reason: Optional[str] = Field(
        default=None, 
        description="Reason for the update decision with information need to updated or added to the graph"
    )
    
    def to_json_string(self):
        """Converts the UpdateGraphDecision object to a JSON string."""
        return json.dumps(self.model_dump(mode='json'), indent=4)
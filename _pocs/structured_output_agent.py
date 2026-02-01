# Key considerations for structured output
# Method parameter: Some providers support different methods for structured output:
# 'json_schema': Uses dedicated structured output features offered by the provider.
# 'function_calling': Derives structured output by forcing a tool call that follows the given schema.
# 'json_mode': A precursor to 'json_schema' offered by some providers. Generates valid JSON, but the schema must be described in the prompt.
# Include raw: Set include_raw=True to get both the parsed output and the raw AI message.
# Validation: Pydantic models provide automatic validation. TypedDict and JSON Schema require manual validation.
# See your provider’s integration page for supported methods and configuration options.

from pydantic import BaseModel, Field

class Movie(BaseModel):
    """A movie with details."""
    title: str = Field(..., description="The title of the movie")
    year: int = Field(..., description="The year the movie was released")
    director: str = Field(..., description="The director of the movie")
    rating: float = Field(..., description="The movie's rating out of 10")

model_with_structure = model.with_structured_output(Movie)
response = model_with_structure.invoke("Provide details about the movie Inception")
print(response)  # Movie(title="Inception", year=2010, director="Christopher Nolan", rating=8.8)

# It can be useful to return the raw AIMessage object alongside the parsed representation to access response metadata such as token counts. 
# To do this, set include_raw=True when calling with_structured_output:

from pydantic import BaseModel, Field

class Movie(BaseModel):
    """A movie with details."""
    title: str = Field(..., description="The title of the movie")
    year: int = Field(..., description="The year the movie was released")
    director: str = Field(..., description="The director of the movie")
    rating: float = Field(..., description="The movie's rating out of 10")

model_with_structure = model.with_structured_output(Movie, include_raw=True)  
response = model_with_structure.invoke("Provide details about the movie Inception")
response
# {
#     "raw": AIMessage(...),
#     "parsed": Movie(title=..., year=..., ...),
#     "parsing_error": None,
# }

# Schemas can be nested:

from pydantic import BaseModel, Field

class Actor(BaseModel):
    name: str
    role: str

class MovieDetails(BaseModel):
    title: str
    year: int
    cast: list[Actor]
    genres: list[str]
    budget: float | None = Field(None, description="Budget in millions USD")

model_with_structure = model.with_structured_output(MovieDetails)
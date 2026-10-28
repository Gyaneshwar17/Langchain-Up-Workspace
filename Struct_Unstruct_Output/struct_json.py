
from pygments.token import Literal
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel, Field
load_dotenv()

chat=ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite',max_tokens=300)
#schema 
DetailedProductReview_json={
  "title": "DetailedProductReview",
  "type": "object",
  "properties": {
    "product_name": {
      "type": "string",
      "description": "Name of the product being reviewed"
    },
    "category": {
      "type": "string",
      "description": "Product category"
    },
    "overall_rating": {
      "type": "number",
      "description": "Overall rating out of 5.0"
    },
    "key_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Write down all the key themes discussed in the review in a list"
    },
    "summary": {
      "type": "string",
      "description": "A concise summary of the review"
    },
    "pros": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "List of distinct advantages"
    },
    "cons": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "List of distinct disadvantages"
    }
  },
  "required": [
    "product_name",
    "category",
    "overall_rating",
    "key_themes",
    "summary",
    "pros",
    "cons"
  ],
  "additionalProperties": False
}

strut_model=chat.with_structured_output(DetailedProductReview_json)

result=strut_model.invoke("""
I've been using the Dell XPS 16 (2025 model) with the OLED touch display and Intel Core Ultra 7 processor for about three weeks now, 
and it is an absolute powerhouse for creative workflows and daily productivity. 

First off, the display is breathtaking. The 4K OLED panel delivers infinite contrast, punchy colors, and deep blacks that make video editing 
and photo grading in Lightroom an absolute joy. Performance-wise, the machine handles multitasking effortlessly. I regularly have dozens of 
Chrome tabs open alongside Docker containers, VS Code, and Blender projects without a single stutter. The keyboard has a satisfying, tactile 
click with comfortable travel, and the seamless glass haptic trackpad feels futuristic and precise.

However, it's not without its drawbacks. The futuristic touch-function row takes some getting used to and lacks the tactile feedback of physical keys. 
Due to its thin profile and powerful internals, the fans ramp up noticeably and the chassis gets quite warm under heavy rendering loads. 
Lastly, port selection is limited mostly to Thunderbolt 4 ports, meaning you will need dongles for legacy USB-A accessories or full-size HDMI cables.

Overall, if you are a developer, designer, or content creator looking for a premium Windows workstation with stunning visuals and stellar build quality, 
this laptop is well worth the investment, provided you don't mind carrying a USB-C hub.
""")

print(type(result),result)

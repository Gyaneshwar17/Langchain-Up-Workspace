from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

prompt=PromptTemplate(
    template="Generate a 5 unique interseting facts about {topic}",
    input_variables=['topic']
)

model=ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite',max_tokens=500)

parser=StrOutputParser()

chain= prompt| model| parser


result=chain.invoke({'topic':'F1'})

print(result)

chain.get_graph().print_ascii()

'''
Here are 5 unique and fascinating facts about Formula 1 that go beyond the basic statistics:

### 1. Drivers Lose Up to 4 Kilograms (8.8 lbs) in a Single Race
Formula 1 drivers are among the fittest athletes in the world, and race days are essentially extreme weight-loss sessions. Due to intense G-forces (up to 5G in corners and braking), cockpit temperatures that can reach 50°C (122°F), and the sheer physical exertion required, drivers can sweat out as much as 3 to 4 kilograms of water weight during a single 90-minute Grand Prix.

### 2. An F1 Steering Wheel is More Complex Than a Fighter Jet's
Modern F1 steering wheels are marvels of engineering, acting as the central command hub for the car. Costing around $50,000 each, they feature up to 25 buttons, switches, and paddles. Drivers can adjust brake balance, differential settings, clutch bite points, radio communication, energy recovery systems (ERS), and even activate the DRS (Drag Reduction System) on the fly—all while traveling at over 300 km/h (186 mph).

### 3. Pit Stops Happen in Under Two Seconds (And Tire Nut Mechanics are Elite Athletes)
The speed of an F1 pit stop defies belief. A team of around 20 mechanics can lift an 800+ kg car, remove all four wheels, put four brand-new ones on, and drop the car back down in **less than 2 seconds**. The current world record is held by McLaren, who changed four tires in a mind-boggling **1.80 seconds** at the 2023 Qatar Grand Prix.

### 4. F1 Brakes Can Glow Red-Hot at 1,000°C (1,832°F)
Carbon-carbon brake discs are used in F1 because they can handle extreme thermal shock. Under heavy braking (such as the end of the straight at the Canadian Grand Prix, where cars slow down from about 320 km/h to 140 km/h in just a couple of seconds), the friction generates temperatures exceeding **1,000°C**. At night races, this causes the brake discs to glow a
      +-------------+      
      | PromptInput |
      +-------------+
             *
             *
             *
    +----------------+
    | PromptTemplate |
    +----------------+
             *
             *
             *
+------------------------+
| ChatGoogleGenerativeAI |
+------------------------+
             *
             *
             *
    +-----------------+
    | StrOutputParser |
    +-----------------+
             *
             *
             *
+-----------------------+
| StrOutputParserOutput |
+-----------------------+

'''
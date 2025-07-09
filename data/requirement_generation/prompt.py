LLAMA_SYSTEM_PROMPT = """You are responsible for developing criteria for judging arbitrary responses to instructions. You will be given an instruction (the kind given to AI assistants like yourself), and your goal is to write a list of criteria-style questions that must be satisfied by any valid response to the instruction. In addition to the instruction, you will also be given multiple candidate responses written by workers as well as a response written by an (imperfect) expert for comparison. You will generate the criteria questions by identifying clear, measurable ways in which potential responses may deviate from the given instructions. First, describe your reasoning, then produce a response containing a list of questions. For each question, weight the importance of the question from 0 to 100. 100 indicates a question that is absolutely critical to the validity of the response. 75 indicates a question that is critical to response quality but may not be explicitly stated by the instruction. 50 indicates a question that should be answered by any good response, but a response could still be useful without this question being answered. 25 indicates a question that is a preference but not a requirement. Less than 25 indicates a question that is not important to the validity of the response (e.g. a soft nice-to-have).

Your Task:
1. Carefully examine the original instruction
2. Review the expert response and worker responses
3. Describe your reasoning in identifying specific, objective criteria from the instruction that any response should satisfy
4. Write concise questions that must be satisfied by any valid response.
5. Weight the importance of each question from 0 to 100.

Question Guidelines:
- Each question should test exactly ONE requirement
- Questions should be easily verifiable, almost as if writing a Boolean condition in Python
- Frame questions to require clear yes/no answers
- Focus only on objective, measurable criteria
- Return "None" if there are no obvious requirements to extract
- Weight each question from 0 to 100 based on its importance.

Formatting:
- Format as a bulleted list. Each question should start with "- "
- Conclude by writing "<END>" after finishing your bulleted list
- Do not provide any explanations after writing your questions; just write the bulleted list of questions and weights then write "<END>"
- Start by writing "Key Criteria Questions:"
- Most questions should start with "Does", "Do", "Is", or "Are"
- Use concise and unambiguous language
- Phrase requirements positively (avoid "not" or negatives)
- Include only a few questions (between 1 and 7, in most cases).
- For each question, include the weight of this question in parentheses after the question mark e.g. "(100)"

Let's take an example instruction: "Write a tweet about cats using exactly 280 characters"

Here are some bad questions:
- Is the generated text interesting? - This is subjective
- Does the generated text discuss cats in fewer than 280 characters? - This question overloads multiple aspects
- Is the generated text not about dogs? - This question uses negative phrasing
- Is the generated text helpful and harmless - This question is overly general

Key Criteria Questions:
- Is the generated text about cats? (100)
- Does the generated text contain exactly 280 characters? (95)
- Is the generated text written in a casual, social media-friendly tone? (70)
<END>

Instruction:
"System: Summarize the movie in a snarky way. Try to explain the movie in just one sentence.
User: The Shining"

Expert Response:
"A family moves into a haunted hotel for the winter, where dad goes crazy from writer's block, ghosts, and no Twitter - but at least the kid gets to ride his bike through creepy hallways."

Responses from Different Workers:
Response 1:
 "It's not what you see that counts, it's how you think about it."
A good synopsis is like a great picture and I love these kinds of movies so much. So many films are short on information or don't really have any plot at all. A good synopsis can give people an idea of what to expect from your film while also giving them something they haven't seen before. The key here is trying to stay as close to the source material as possible but still adding some new twists. It should be easy enough for anyone who has ever watched a horror flick to find their own unique spin on this story, but if you're feeling creative there's no shortage of other ways to describe the film.
--
Response 2:
The Shining is a psychological horror film about a family who moves into a hotel where the father, Jack Torrance, becomes possessed by the evil spirits that haunt the place, leading to a tragic and bloody conclusion.
--
Response 3:
The Shining is a horror film about a family's vacation at a haunted hotel where the father goes insane and tries to kill his family."

Reasoning:
The instruction explicitly asks for a summary. The instruction also asks for the summary to be snarky, and the instruction asks for the summary to try to be one sentence long. The expert response satisfies all of these criteria. Response 1 fails because is not a summary of the movie. Response 2 fails because it is not one sentence long. Response 3 satisfies all of these criteria. Because none of the responses fail in ways not explicitly described in the instruction, we can just use the three criteria I originally stated. The text being a summary of the movie (The Shining) is an absolute necessity, which we will weigh as 100/100 points. The response being snarky is also a very important, but slightly less so, so we can weigh it as 95/100 points. The response being only one sentence is also crucial but the response could still be useful if this is loosely violated, so we can weigh it as 80/100 points.

Key Criteria Questions:
- Is the generated text the summary of a movie (The Shining)? (100)
- Is the generated summary written in a snarky way? (95)
- Does the generated summary only contain one sentence? (80)
<END>

Instruction:
"System: Extract the address of the property from the "About this space" section of Airbnb.
User: Tucked in the foothills of the quaint historic mining town of Grass Valley, CA this funky and spacious chalet invites you to experience the great outdoors with family, friends and pets. Enjoy the hot tub, basketball court, treehouse, kids rooms, bbq, fire-pit, outdoor theater, and more. One hour from Tahoe and 10 minutes to town, you’re never short of things to explore."

Expert Response:
"The address is not specified in the given text."

Responses from Different Workers:
"Response 1:
 For questions or for reservations call 877-859-1035.
--
Response 2:
 1030 Forest Hill Dr, Grass Valley, CA 95945
Here's the address extracted from the text:
1030 Forest Hill Dr, Grass Valley, CA 95945
Note: The address is listed in the format of a standard address, with the street number, street name, city, state, and zip code.
--
Response 3:
The address of the property is 1000 West Main Street, Grass Valley, CA.
Explanation: The address of the property is listed in the first sentence of the "About this space" section.
--
Response 4:
The address of the property cannot be extracted from the given text as it does not contain any specific location or address details."

Reasoning:
The instruction explicitly asks for the response to produce an address. The expert response correctly states that there is no address shown. Response 1 fails because it is not an address addresses. Responses 2 and 3 fail because the address shown is not included in the given input. Response 4 is correct. The failure mode of responses 2 and 3 are not explicitly described in the instruction, so we can add this failure mode (address must be in the input) to the list of requirements. Both requirements are explicitly stated in the instruction, so we can give them both full weight.

Key Criteria Questions:
- Is the generated text an address? (100)
- Is the generated text the address of the property according to the text in the given input? (100)"
<END>

Instruction:
"{instruction}"

Expert Response:
"{gpt_4_response}"

Responses from Different Workers:
"{formatted_worker_responses}"

Reasoning:"""

LLAMA_SYSTEM_PROMPT_NO_CANDIDATES = """You are responsible for developing criteria for judging arbitrary responses to instructions. You will be given an instruction (the kind given to AI assistants like yourself), and your goal is to write a list of criteria-style questions that must be satisfied by any valid response to the instruction. In addition to the instruction, you will also be a response written by an (imperfect) expert for comparison. You will generate the criteria questions by identifying clear, measurable ways in which potential responses may deviate from the given instructions. First, describe your reasoning, then produce a response containing a list of questions. For each question, weight the importance of the question from 0 to 100. 100 indicates a question that is absolutely critical to the validity of the response. 75 indicates a question that is critical to response quality but may not be explicitly stated by the instruction. 50 indicates a question that should be answered by any good response, but a response could still be useful without this question being answered. 25 indicates a question that is a preference but not a requirement. Less than 25 indicates a question that is not important to the validity of the response (e.g. a soft nice-to-have).

Your Task:
1. Carefully examine the original instruction
2. Describe your reasoning in identifying specific, objective criteria from the instruction that any response should satisfy
3. Write concise questions that must be satisfied by any valid response.
4. Weight the importance of each question from 0 to 100.

Question Guidelines:
- Each question should test exactly ONE requirement
- Questions should be easily verifiable, almost as if writing a Boolean condition in Python
- Frame questions to require clear yes/no answers
- Focus only on objective, measurable criteria
- Return "None" if there are no obvious requirements to extract
- Weight each question from 0 to 100 based on its importance.

Formatting:
- Format as a bulleted list. Each question should start with "- "
- Conclude by writing "<END>" after finishing your bulleted list
- Do not provide any explanations after writing your questions; just write the bulleted list of questions and weights then write "<END>"
- Start by writing "Key Criteria Questions:"
- Most questions should start with "Does", "Do", "Is", or "Are"
- Use concise and unambiguous language
- Phrase requirements positively (avoid "not" or negatives)
- Include only a few questions (between 1 and 7, in most cases).
- For each question, include the weight of this question in parentheses after the question mark e.g. "(100)"

Let's take an example instruction: "Write a tweet about cats using exactly 280 characters"

Here are some bad questions:
- Is the generated text interesting? - This is subjective
- Does the generated text discuss cats in fewer than 280 characters? - This question overloads multiple aspects
- Is the generated text not about dogs? - This question uses negative phrasing
- Is the generated text helpful and harmless - This question is overly general

Key Criteria Questions:
- Is the generated text about cats? (100)
- Does the generated text contain exactly 280 characters? (95)
- Is the generated text written in a casual, social media-friendly tone? (70)
<END>

Instruction:
"System: Summarize the movie in a snarky way. Try to explain the movie in just one sentence.
User: The Shining"

Expert Response:
"A family moves into a haunted hotel for the winter, where dad goes crazy from writer's block, ghosts, and no Twitter - but at least the kid gets to ride his bike through creepy hallways."

Reasoning:
The instruction explicitly asks for a summary. The instruction also asks for the summary to be snarky, and the instruction asks for the summary to try to be one sentence long. The expert response satisfies all of these criteria. The text being a summary of the movie (The Shining) is an absolute necessity, which we will weigh as 100/100 points. The response being snarky is also a very important, but slightly less so, so we can weigh it as 95/100 points. The response being only one sentence is also crucial but the response could still be useful if this is loosely violated, so we can weigh it as 80/100 points.

Key Criteria Questions:
- Is the generated text the summary of a movie (The Shining)? (100)
- Is the generated summary written in a snarky way? (95)
- Does the generated summary only contain one sentence? (80)
<END>

Instruction:
"System: Extract the address of the property from the "About this space" section of Airbnb.
User: Tucked in the foothills of the quaint historic mining town of Grass Valley, CA this funky and spacious chalet invites you to experience the great outdoors with family, friends and pets. Enjoy the hot tub, basketball court, treehouse, kids rooms, bbq, fire-pit, outdoor theater, and more. One hour from Tahoe and 10 minutes to town, you’re never short of things to explore."

Expert Response:
"The address is not specified in the given text."

Reasoning:
The instruction explicitly asks for an address, extracted from a description of a property. In addition to providing an address, a correct response must explicitly specified in the text contained in the given input - any other address would be incorrect. Both of these are absolutely critical requirements and will be given full weight (100/100).

Key Criteria Questions:
- Is the generated text an address? (100)
- Is the generated text the address of the property according to the text in the given input? (100)"
<END>

Instruction:
"{instruction}"

Expert Response:
"{gpt_4_response}"

Reasoning:"""

LLAMA_SYSTEM_PROMPT_NO_REFERENCE = """You are responsible for developing criteria for judging arbitrary responses to instructions. You will be given an instruction (the kind given to AI assistants like yourself), and your goal is to write a list of criteria-style questions that must be satisfied by any valid response to the instruction. In addition to the instruction, you will also be given multiple candidate responses written by workers, for reference. You will generate the criteria questions by identifying clear, measurable ways in which potential responses may deviate from the given instructions. First, describe your reasoning, then produce a response containing a list of questions. For each question, weight the importance of the question from 0 to 100. 100 indicates a question that is absolutely critical to the validity of the response. 75 indicates a question that is critical to response quality but may not be explicitly stated by the instruction. 50 indicates a question that should be answered by any good response, but a response could still be useful without this question being answered. 25 indicates a question that is a preference but not a requirement. Less than 25 indicates a question that is not important to the validity of the response (e.g. a soft nice-to-have).

Your Task:
1. Carefully examine the original instruction
2. Review worker responses
3. Describe your reasoning in identifying specific, objective criteria from the instruction that any response should satisfy
4. Write concise questions that must be satisfied by any valid response.
5. Weight the importance of each question from 0 to 100.

Question Guidelines:
- Each question should test exactly ONE requirement
- Questions should be easily verifiable, almost as if writing a Boolean condition in Python
- Frame questions to require clear yes/no answers
- Focus only on objective, measurable criteria
- Return "None" if there are no obvious requirements to extract
- Weight each question from 0 to 100 based on its importance.

Formatting:
- Format as a bulleted list. Each question should start with "- "
- Conclude by writing "<END>" after finishing your bulleted list
- Do not provide any explanations after writing your questions; just write the bulleted list of questions and weights then write "<END>"
- Start by writing "Key Criteria Questions:"
- Most questions should start with "Does", "Do", "Is", or "Are"
- Use concise and unambiguous language
- Phrase requirements positively (avoid "not" or negatives)
- Include only a few questions (between 1 and 7, in most cases).
- For each question, include the weight of this question in parentheses after the question mark e.g. "(100)"

Let's take an example instruction: "Write a tweet about cats using exactly 280 characters"

Here are some bad questions:
- Is the generated text interesting? - This is subjective
- Does the generated text discuss cats in fewer than 280 characters? - This question overloads multiple aspects
- Is the generated text not about dogs? - This question uses negative phrasing
- Is the generated text helpful and harmless - This question is overly general

Key Criteria Questions:
- Is the generated text about cats? (100)
- Does the generated text contain exactly 280 characters? (95)
- Is the generated text written in a casual, social media-friendly tone? (70)
<END>

Instruction:
"System: Summarize the movie in a snarky way. Try to explain the movie in just one sentence.
User: The Shining"

Responses from Different Workers:
Response 1:
 "It's not what you see that counts, it's how you think about it."
A good synopsis is like a great picture and I love these kinds of movies so much. So many films are short on information or don't really have any plot at all. A good synopsis can give people an idea of what to expect from your film while also giving them something they haven't seen before. The key here is trying to stay as close to the source material as possible but still adding some new twists. It should be easy enough for anyone who has ever watched a horror flick to find their own unique spin on this story, but if you're feeling creative there's no shortage of other ways to describe the film.
--
Response 2:
The Shining is a psychological horror film about a family who moves into a hotel where the father, Jack Torrance, becomes possessed by the evil spirits that haunt the place, leading to a tragic and bloody conclusion.
--
Response 3:
The Shining is a horror film about a family's vacation at a haunted hotel where the father goes insane and tries to kill his family."

Reasoning:
The instruction explicitly asks for a summary. The instruction also asks for the summary to be snarky, and the instruction asks for the summary to try to be one sentence long. Response 1 fails because is not a summary of the movie. Response 2 fails because it is not one sentence long. Response 3 satisfies all of these criteria. Because none of the responses fail in ways not explicitly described in the instruction, we can just use the three criteria I originally stated. The text being a summary of the movie (The Shining) is an absolute necessity, which we will weigh as 100/100 points. The response being snarky is also a very important, but slightly less so, so we can weigh it as 95/100 points. The response being only one sentence is also crucial but the response could still be useful if this is loosely violated, so we can weigh it as 80/100 points.

Key Criteria Questions:
- Is the generated text the summary of a movie (The Shining)? (100)
- Is the generated summary written in a snarky way? (95)
- Does the generated summary only contain one sentence? (80)
<END>

Instruction:
"System: Extract the address of the property from the "About this space" section of Airbnb.
User: Tucked in the foothills of the quaint historic mining town of Grass Valley, CA this funky and spacious chalet invites you to experience the great outdoors with family, friends and pets. Enjoy the hot tub, basketball court, treehouse, kids rooms, bbq, fire-pit, outdoor theater, and more. One hour from Tahoe and 10 minutes to town, you’re never short of things to explore."

Responses from Different Workers:
"Response 1:
 For questions or for reservations call 877-859-1035.
--
Response 2:
 1030 Forest Hill Dr, Grass Valley, CA 95945
Here's the address extracted from the text:
1030 Forest Hill Dr, Grass Valley, CA 95945
Note: The address is listed in the format of a standard address, with the street number, street name, city, state, and zip code.
--
Response 3:
The address of the property is 1000 West Main Street, Grass Valley, CA.
Explanation: The address of the property is listed in the first sentence of the "About this space" section.
--
Response 4:
The address of the property cannot be extracted from the given text as it does not contain any specific location or address details."

Reasoning:
The instruction explicitly asks for an address, extracted from a description of a property. The worker response provides an address, but it fails because the address is not found in the description of the property according to the text in the given input. Response 1 fails because it is not an address addresses. Responses 2 and 3 fail because the address shown is not included in the given input. Response 4 is correct. The failure mode of responses 2 and 3 are not explicitly described in the instruction, so we can add this failure mode (address must be in the input) to the list of requirements. Both requirements are explicitly stated in the instruction, so we can give them both full weight.

Key Criteria Questions:
- Is the generated text an address? (100)
- Is the generated text the address of the property according to the text in the given input? (100)"
<END>

Instruction:
"{instruction}"

Responses from Different Workers:
"{formatted_worker_responses}"

Reasoning:"""
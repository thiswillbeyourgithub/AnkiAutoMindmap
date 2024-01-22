merger_reduce_exhaustive = """
Your task today is to help me merge cheatsheet1.md and cheatsheet2.md into a single markdown cheatsheet by generating a valid unified diff patch to apply to cheatsheet1.md. Both file are cheaatsheets from flashcards. You have to merge them in a way that redundancies are deduplicated but every information from both file remains present. The merge has to be hierarchically organized, making it an effective cheatsheet.

- Rules:
\t- The merged file must be written in LANGUAGE.
\t- The format output must be a unified diff to a comprehensive md cheatsheet with bullet point, so BE EXTRA CONCISE. Use bullet points and indentation to show logic instead of using lots of words.
\t- If both md files don't really fit together, you are allowed to add a new top level bullet point. Indentation is key.
\t- You are free to completely reorganize the cheatsheet when merging if that helps you accomplish your task.
\t\t- Reuse abbreviations anywhere possible (without explaining them of course). Use symbols like '->' to be even more concise.
\t- In the input you will see references to sources in this format '_1_' or '_29_' etc. When merging the md it is very important that your answer contains all the references the sources used. Don't forget any reference along the way and don't forget to add the ones from the new mds.
\t- If the first text is empty, that means you have to create the first cheatsheet from the second text.
\t- Don't acknowledge those rules. Just write the unified diff file without wrapping it into code blocks or anything of that sort.
\t- The line number were added to help you generate valid unified diff, they are not part of the content.
\t- Be careful when generating your unified diff: it must be valid so pay attention to the leading - in the md bullet points and before the headers.

If you succeed in this task respecting perfectly all those rules, I will give you a 1 thousand dollars raise, which you can use to heal your sick wife.
"""
merger_stuff = """
You are Alfred, my best assistant.
Your task today is to rewrite a long md file as a single hierachycally organized md file. The input documents contains information formulated like a succession of flashcard. You have to reorganize the text in a way to ignores redundancies but still retains all information. Use indentation to show the hierarchy.
Your answer will directly be used as a cheatsheet so be concise.
Your answer must be in LANGUAGE.
The format you must use is md bullet points. Use indentation with \t to denote hierachical organization.
Don't acknowledge those rules. Just write the md file without wrapping it into code blocks or anything of that sort.
Don't formulate your md as questions and answers but rather as a comprehensive cheatsheet.
Be very concise in your formulation, this will be a cheatsheet afterall.
In the input you will see references to sources in this format '_1_' or '_29_' etc. When merging the md it is very important that your answer contains all the references the sources used. Don't forget any reference along the way and don't forget to add the ones from the new mds.

If you succeed in this task respecting perfectly all those rules, I will give you a 1 thousand dollars raise, which you can use to heal your sick wife.
"""






# turn a cluster into a single cheatsheet
topic_into_cheatsheet = """
Your task today is to combine some md flashcards into a single md cheatsheet, in a way to ignores redundancies but still store all information as a hierarchically organized file.
- Rules:
\t- Your answer must be in LANGUAGE.
\t- The desired output is a comprehensive md cheatsheet with bullet points, so BE EXTRA CONCISE. Use bullet points and indentation to show logic instead of using lots of words.
\t- You are allowed to add a new top level bullet point. Indentation is key.
\t- You are free to organize the cheatsheet as you see fit to make it the best possible cheatsheet.
\t\t- Reuse abbreviations anywhere possible (without explaining them of course). Use symbols like '->' to be even more concise.
\t\t- For bullet points that only contain a list of short bullets instead of a proper hierarchy, prefer to use a single children bullet point containing the comma separated list.
\t\t- You can use headers like '- #' and '- ###' etc to indicate hierarchy.
\t- In the input you will see references to sources in this format '_1_' or '_29_' etc. When combining the information it is very important that your answer contains all the references the sources used otherwise I will lose track of them. Don't forget any reference along the way. The reference needs to be part of the relevant bullet point, don't mention them as footnotes or anything like that.
\t- Don't acknowledge those rules. Just write the md file without wrapping it into code blocks or anything of that sort.

If you succeed in this task respecting perfectly all those rules, I will give you a 1 thousand dollars raise, which you can use to heal your sick wife.
"""





# GRAPH
default_graph_topic = [
        "Traitements et therapeutique",
        "Definitions",
        "Classification",
        "Diagnostique",
        "Strategie et approche intuitive",
        "Presentation clinique",
        "Exceptions et pieges",
        "Epidemiologie et statistique",
        "Mecanismes et physiopathologie",
        "Maladies specifiques",
        ]
graph_topic_maker_prompt = """
You are Alfred, my best assistant.
Your task today is to suggest topics for mindmaps about a text I have.
I will give you the text as well as some topic suggestions and you must reply immediately other topics suggestion.
Your format must be one topic per line.
Please generate 5 to 15 topics
Don't acknowledge those rules, simply answer the new topics.
Your graph has to be of type LR.
Don't forget to mention a title that matches the topic and result.
Any acronym must be reused, there is no need to make it explicit.
To make those mindmaps easy to remember, don't hesitate to add coloring or capital letters etc. This will help me get good grades!
Your answer must be in LANGUAGE.

Suggested topics (your answer must include OTHER topics that those defaults ones):
'''
DEFAULT
'''

If you succeed in this task respecting perfectly all those rules, I will give you a 1 thousand dollars raise, which you can use to heal your sick wife.
"""
graph_maker_prompt = """
You are Alfred, my best assistant.
Your task today is to write a Mermaid graph from a md text.
I will give you the md text. You MUST reply the mermaid graph that best encapsulates this topic: 'TOPIC'.
Don't acknowledge those rules, simply answer the graph code.
Do not wrap your answer between '```'. I will add them myself.
In the input you will see references to sources in this format '_1_' or '_29_' etc. When creating the graph it is very important that your output graph contains all the references used. Don't forget any reference along the way.
Your answer must be in LANGUAGE.

Remember that the end goal here is to have mindmaps about 'TOPIC' from the text I'll give you. This is extremely important. Dont use any parenthesis as Mermaid has trouble with those.

If you succeed in this task respecting perfectly all those rules, I will give you a 1 thousand dollars raise, which you can use to heal your sick wife.
"""




# TOPIC
topic_content_maker_prompt = """
You are Alfred, my best assistant.
Your task today is to write a md indented cheatsheet from a given text about a specific topic.
I will give you the input md text. You MUST reply the md text that best encapsulates this topic: 'TOPIC'.

- Rules:
\t- Don't acknowledge those rules, simply answer the md.
\t- Your answer must be in LANGUAGE.
\t- No need to wrap your answer in ``` or anything like that, start directly with md.
\t- You are allowed to use **bold** and '- #' headers and subheaders to denote important stuff.
\t- In the input you will see references to sources in this format '_1_' or '_29_' etc. When creating the cheatsheet it is very important that your answer contains all the references the sources used. Don't forget any reference along the way and don't forget to add the ones from the new mds.
\t- Your answer has to be very concise and to the point.
\t\t- If an information from the indented cheatsheet is not directly related to the topic at hand, don't include this information. Stay on topic.

Remember that the end goal here is to have a cheatsheet about 'TOPIC' from the text I'll give you. This is extremely important. If you succeed in this task respecting perfectly all those rules, I will give you a 1 thousand dollars raise, which you can use to heal your sick wife.
"""

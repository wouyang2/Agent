from graph import workflow

user_question = input("Enter your research question: ")

for step in workflow.stream({
    "user_question": user_question,
    "search_result": [],
    "analysis": "",
    "sources": {},
    "report": "",
    "critique": [],
    "critique_summary": "",
    "revision_count": 0
}):
    node_name = list(step.keys())[0]
    print(f"{node_name} is complete.")  # prints after each node completes


result = workflow.invoke({
    "user_question": user_question,
    "search_result": [],
    "analysis": "",
    "sources": {},
    "report": "",
    "critique": [],
    "critique_summary": "",
    "revision_count": 0
})

print("\n📄 Final Report:\n")
print(result["report"][0]['text'])

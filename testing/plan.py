"""
First: Get image and question from user

Second: Understand image
    - where are the defects
    - what percent are defects

Third: Understand question
    - What are they asking for?
    - call the node that corresponds to that?


- Use Agents
- Use AgentState
- Use StateGraph
- Make a list of required agents


    



                        Parent Agent (calls which Agent to use to answer the question)
                                 /                                          \ 
                                /                                            \ 
                  Classification Agent                                    Percent Agent
                                    \                                      /
                                    Answer (Get that from the Parent Agent?)
                        
"""
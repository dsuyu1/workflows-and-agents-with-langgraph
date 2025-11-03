# Extending Workflows
In the previous section, I created a workflow in LangGraph that begins by classifying the user's ticket, retrieving knowledge from the knowledge base, drafting a response, evaluating that response, and revising it if neccessary. 

There were a few issues, however. Notably, the model's final draft was:

> For login issues, tell the user to try resetting their password via the 'Forgot Password' link. The app is known to crash on startup if the user's cache is corrupted. The standard fix is to clear the application cache. Billing inquiries should be escalated to the billing department by creating a ticket in Salesforce.

It basically just regurgitated the information found in the knowledge base. 

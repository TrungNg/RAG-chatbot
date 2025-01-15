import json
import boto3
import os

bedrock = boto3.client('bedrock-runtime')
retriever = boto3.client('bedrock-agent-runtime')

def lambda_handler(event, context):
    try:
        # Check if the event is coming from API Gateway (has 'body' key) or direct invocation
        if 'body' in event:
            # API Gateway format
            body = json.loads(event['body'])
        else:
            # Direct invocation format
            body = event

        messages = body.get('messages', [])
        max_tokens = int(body.get('max_tokens', 1024))
        
        # Extract the last user message as the query
        query = next((msg['content'] for msg in reversed(messages) if msg['role'] == 'user'), None)
        
        if not query:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No user message found'})
            }
        
        ## Placeholder: divide user's queries when the topic is changed.
        ## Placeholder: test whether using only the last query for embedding/RAG

        # Retrieve relevant information from the knowledge base
        retrieve_response = retriever.retrieve(
            knowledgeBaseId=os.environ['KNOWLEDGE_BASE_ID'],
            retrievalQuery={
                'text': query
            },
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': 3
                }
            }
        )

        reference_data = []
        for idx, passage in enumerate(retrieve_response['retrievalResults']):
            reference_data.append({
                'title': passage.get('title', f'Reference #{idx+1}'),
                'text': passage['content']['text'] if 'text' in passage['content'] else passage['content']
            })

        # Construct prompt for the LLM
        system_message = "You are a helpful assistant. Use the following referenced information in combination with your knowledge to answer the user's question. Cite the references with their corresponding numbers, e.g. \[2\] for Reference number 2, in your answer if relevant."
        context_message = "\n".join(f"{passage['title']}\n{passage['text']}" for passage in reference_data)
        
        ## Placeholer: reorganize the 'messages' field of claude_messages to retain the chat history
        
        # Prepare messages for Claude
        claude_messages = [
            {"role": "user", "content": f"{system_message}\n\nReferences:\n{context_message}\n\nUser query: {query}"}
        ]

        ## Placeholder: extract model names, convert popular model names to their AWS ARN/ID

        # Invoke Bedrock model (Sonnet v2)
        response = bedrock.invoke_model(
            modelId='anthropic.claude-3-haiku-20240307-v1:0', #'arn:aws:bedrock:ap-southeast-2:148761647764:inference-profile/apac.anthropic.claude-3-5-sonnet-20240620-v1:0', 
            contentType='application/json',
            accept='application/json',
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": claude_messages,
                "temperature": body.get('temperature', 0.7),
                "top_p": body.get('top_p', 1)
            })
        )

        # Parse the response
        response_body = json.loads(response['body'].read())
        generated_text = response_body['content'][0]['text']

        # Return the response
        return {
            'statusCode': 200,
            'body': json.dumps({
                'response': generated_text,
                'model': 'claude-3-sonnet-20240620-v1',
                'references': reference_data  # <-- Include references here
            })
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Internal Server Error'})
        }
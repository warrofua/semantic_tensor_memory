#!/usr/bin/env python3
"""
Convert test_conversation.json to ChatGPT export format expected by the app.
"""

import json
import time
from datetime import datetime
from typing import Dict, Any

def convert_to_chatgpt_format(input_file: str, output_file: str):
    """
    Convert our custom conversation JSON to ChatGPT export format.
    
    Expected ChatGPT format:
    {
        "id": "conversation_id",
        "conversation_id": "conversation_id", 
        "mapping": {
            "msg_id_1": {
                "message": {
                    "content": {"parts": ["text content"]},
                    "author": {"role": "user"},
                    "create_time": timestamp
                }
            },
            "msg_id_2": {
                "message": {
                    "content": {"parts": ["text content"]},
                    "author": {"role": "assistant"},
                    "create_time": timestamp
                }
            }
        }
    }
    """
    
    # Load our conversation JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        conversation_data = json.load(f)
    
    # Create ChatGPT format structure
    conv_id = "linkedin_optimization_conversation"
    base_timestamp = time.time()
    
    chatgpt_format = {
        "id": conv_id,
        "conversation_id": conv_id,
        "title": conversation_data.get("title", "Conversation"),
        "create_time": base_timestamp,
        "mapping": {}
    }
    
    # Convert each message to ChatGPT format
    for i, message in enumerate(conversation_data["messages"]):
        msg_id = f"msg_{i:04d}"
        
        # Convert role mapping
        role = message["role"]
        if role == "assistant":
            role = "assistant"
        else:
            role = "user"
        
        # Create message structure
        msg_content = message["content"]
        
        # Handle attachments by appending them to content
        if "attachments" in message:
            for attachment in message["attachments"]:
                msg_content += f"\n\n[File: {attachment['description']}]"
        
        chatgpt_message = {
            "message": {
                "id": msg_id,
                "content": {
                    "content_type": "text",
                    "parts": [msg_content]
                },
                "author": {
                    "role": role,
                    "name": None,
                    "metadata": {}
                },
                "create_time": base_timestamp + i,  # Increment timestamp for each message
                "status": "finished_successfully",
                "end_turn": None,
                "weight": 1.0,
                "metadata": {},
                "recipient": "all"
            },
            "parent": f"msg_{i-1:04d}" if i > 0 else None,
            "children": [f"msg_{i+1:04d}"] if i < len(conversation_data["messages"]) - 1 else []
        }
        
        chatgpt_format["mapping"][msg_id] = chatgpt_message
    
    # Write the ChatGPT format JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chatgpt_format, f, indent=2, ensure_ascii=False)
    
    return chatgpt_format

def main():
    """Main function to convert to ChatGPT format."""
    input_file = "test_conversation.json"
    output_file = "test_conversation_chatgpt.json"
    
    try:
        print(f"📖 Reading conversation from {input_file}...")
        result = convert_to_chatgpt_format(input_file, output_file)
        
        message_count = len(result["mapping"])
        print(f"✅ Successfully converted {message_count} messages to ChatGPT format")
        print(f"💾 Saved as {output_file}")
        
        # Validate the format
        print(f"\n🔍 Validation:")
        print(f"   📋 Title: {result['title']}")
        print(f"   🆔 ID: {result['id']}")
        print(f"   💬 Messages: {message_count}")
        print(f"   🗂️ Has mapping: {'✅' if 'mapping' in result else '❌'}")
        
        # Show sample message structure
        first_msg_id = list(result["mapping"].keys())[0]
        first_msg = result["mapping"][first_msg_id]
        print(f"   📝 Sample role: {first_msg['message']['author']['role']}")
        print(f"   📄 Sample content length: {len(first_msg['message']['content']['parts'][0])} chars")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 
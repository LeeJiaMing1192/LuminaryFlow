import discord

# Replace 'YOUR_BOT_TOKEN' with your actual bot token
TOKEN = 'MTM2NzA4NDM5NTExNDAwODY1OA.Gv9UQ-.EgARGhCOPawa9UOB2AGBUX9-MTJ-iQt0T00I3Q'

# Define the intents for your bot. For message content, you need to specify it.
intents = discord.Intents.default()
intents.message_content = True

# Create a client instance
client = discord.Client(intents=intents)

# Event that runs when the bot is ready
@client.event
async def on_ready():
    print(f'Logged in as {client.user}')
    print('Bot is ready to receive Direct Messages.')

# Event that runs when a message is received
@client.event
async def on_message(message):
    # Ignore messages sent by the bot itself to prevent loops
    if message.author == client.user:
        return

    # Check if the message is a direct message
    if isinstance(message.channel, discord.DMChannel):
        sender = message.author
        content = message.content
        print(f"[{message.created_at.strftime('%Y-%m-%d %H:%M:%S')}] Received a DM from {sender} (ID: {sender.id}): {content}")

        # --- Here is where you would integrate with your LLM ---
        # For this basic example, we'll just echo the message back
        llm_response = f"My LLM brain processed your DM: '{content}' sent by {sender.name}!"
        await message.channel.send(llm_response)
        # --- End of LLM integration placeholder ---

    else:
        # This part handles messages in server channels
        # You can add commands or other server-related logic here if needed
        if message.content.startswith('!hello'):
            await message.channel.send(f'Hello {message.author.name}!')

# Run the bot
client.run(TOKEN)
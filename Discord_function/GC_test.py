import discord

# Replace 'YOUR_BOT_TOKEN' with your actual bot token
TOKEN = 'MTM2NzA4NDM5NTExNDAwODY1OA.Gv9UQ-.EgARGhCOPawa9UOB2AGBUX9-MTJ-iQt0T00I3Q'

# Intents tell Discord what events your bot needs to subscribe to.
# For receiving messages, you'll need the 'message_content' intent.
# In newer versions of discord.py, you need to explicitly declare intents.
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'Logged in as {client.user}')

@client.event
async def on_message(message):
    # Ignore messages sent by the bot itself to prevent infinite loops
    if message.author == client.user:
        return

    # The message content
    content = message.content
    # The user who sent the message
    author = message.author
    # The user's display name (nickname if set in the server, otherwise username)
    display_name = message.author.display_name
    # The channel where the message was sent
    channel = message.channel
    # The server (guild) where the message was sent (will be None in DMs)
    guild = message.guild

    if guild:
        print(f"Received message in {guild.name} - #{channel.name} from {display_name} ({author.id}): {content}")
    else:
        print(f"Received private message from {display_name} ({author.id}): {content}")

    # You can now process the message content and respond accordingly
    if content.lower() == 'hello':
        await message.channel.send(f'Hello {display_name} in {channel.name}!')

client.run(TOKEN)
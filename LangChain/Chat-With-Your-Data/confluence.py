from langchain.document_loaders import ConfluenceLoader

loader = ConfluenceLoader(
    # URL should be wihtout /wiki/home at the end
    url="<my url>", 
    # user name should be with @ your domain at the end
    username="<my user>", 
    api_key="<my token>"
)
# space id should not be the display name
documents = loader.load(space_key="<space id>", include_attachments=True, limit=50)

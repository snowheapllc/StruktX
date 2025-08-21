# Smart Device Control with StruktX

This is a sample of a full handler implementation with all its requirements, it fetches the given users devices from a `base_url` according to the specified methods in the `toolkit.py` and `handler.py` which inherits from the base class of Handlers in StruktX.

Provided all the needed transports like signing and also validation of outputs from the LLM, or any other extra steps needed, all can be defined and added to the operation of the handler.

This example focuses on LifeSmart devices.

# Customization

To modify this example, change the `base_url` to your actual service that gets the users device and implement your fetching logic and transport and validation logic (if needed) as well as your prompts for the devices and examples if required too.

Then modify `handler.py` for your own tool / handler logic.

It can be taken apart completely and made specifically for your case as it is defined with plain Pydantic classes inherited from StruktX interfaces and ABC classes.
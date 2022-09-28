# this was just a lot of ideas, and i wanted a place to write it down. A journal of sorts i guess.

# okay, the first one is a quick function / utility. 
# this if made super easy (import sketch as sk; sk.Init(sk.openAI('1234'))) 
# then you can do df.groupby('state').apply(sk.prompt("Colors for NBA fans in Zion: [0xF12C2FE, 0x012345]\nColors for NBA fans in {{ state }}:"))
# ## sk.prompts have a "clean" built in, that ensures structure is right. In this example, it infers that it should be a list of colors (in hex)
# sk.__get_total_number_of_users(..)

# not all prompts work. There is a "fee" on openAI per fill-in, so there's a billing thing there
# can offer "Reduced billing" (if it's known that most prompts that come in can be run as batches)
# eg. those that are run behind pandas like this, are more likely to have a "parallelizable" "apply" structure.
# those (i think) are cheaper to run - based on the "run inference as fast as possible" libraries which have "batch sized based scaling" aparently (linear in that is 'almost free', it's a trade-off dimension?)
# so, if you have a prompt that is "batchable" (like this one) you can get a discount on the "fill-in" cost.

# so, the next one is a "fitbit" dashboard.
# this is a bit more complicated, but it's a good example of how to use the "sketch" library to build a dashboard. -> edit / append to, a dashboard

# could lean into the idea of "sketch" (draw on top of, add a sketch-pad on top, that's a 'digital sandbox' that you can move and put shapes in?? lol, im stretching here, doesn't work really)
import robotic as ry
import numpy as np

C = ry.Config()
C.addFile("$RAI_PATH/scenarios/pandaSingle.g")
bot = ry.BotOp(C, useRealRobot=False)

q = bot.get_q()

print(q.shape)

bot.move([q+.1], times=[bot.get_t()+1/15], overwrite=True)

bot.wait(C)

print(bot.get_q())
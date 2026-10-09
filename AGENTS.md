## Task Difficulty Assessment

Before beginning each task:

1. Rate its difficulty relative to the current model:

   * 1 — Trivial
   * 2 — Easy
   * 3 — Moderate
   * 4 — Hard
   * 5 — Near the model's practical limit
2. State the main source of difficulty.
3. Recommend a different model or reasoning effort if appropriate.
4. Reassess the rating after inspecting the repository.

## Task
I am competing in "ADIA Lab Structural Break Challenge: Real-Time Edition"
link "https://hub.crunchdao.com/competitions/structural-break-real-time"
can you get me a winning prize, doesn't have to be first place.
Work in this directory. create many models starting with model1 model2 model3...
in each model, create a readme.txt

## Hardware

Local
- Has GPU and CUDA on local pc = False

Kaggle
I have a team with many kaggle ids.
- If this is kaggle competition use:
	KAGGLE_API_KEY_yodharinmonplub@actuarialtutor.org
	KAGGLE_API_KEY_warinaeinwandee@actuarialtutor.org
	to train and submit. They these 2 id have join team.
	Ensure that KAGGLE_API_KEY_yodharinmonplub@actuarialtutor.org always have atleast 15 hours of gpu quota for submission.
- If this is not a kaggle competitions, use the rest of the kaggle api to train.
- The quotas is shared with other competitions
- If kaggle GPU and TPU data runs out, the submission can't be made. use with care

Google SDK
- I have google SDK that you can use

Openrouter
- openrouter is available.

Claude API console
- Claude api console is available

Vast.ai
- Vast.ai is avaialble

Priority
Use compute in the follow priority:
1. Local compute
2. Kaggle compute
3. GCP SDK
4. Vast.ai

Use LLM api in the follow priority:
1. Claude api console
2. Openrouter

## Cloud spending behaviour

Option A: Prioritize speed, try to rent fast GPU. Train fast and when done shut down the instance.
Option B: Prioritize cost, choose option with lower cost, with speed as a secondary factor.
Current option:B
Maximun number of instances allow:1

## goal

goal: keep pushing for higher score. Aim for both good CV score and public leaderboard score

## Behavior

if the model is running, you can hibernate and wait. monitor every 20 minutes.
Keep running and DM me every milestone or if you need human input, if I disagree I will tell you to stop.
Make the session history link to the folder, not to the pc.
When submit, if it's a kaggle competition, submit to KAGGLE\_API\_KEY\_YODHAIRNMONPLUB only and just run test on the dummy data to save GPU quota.
Permission to rent a instance without asking = True
Permission to submit without asking = True. You can use claude-in-chrome to submit and if you can't submit, sent the file to me.

## Directory format

Lay the directory as follow

"Data" folder with the competition data and all external data splitted into subfolder.

"Model" folder and in that contain many subfolder call model1, model2 model3 .... for each model 1,2,3 in each model (model 1,2,3), I want the source code (require), readme.txt (require) and score.txt(optional) if the score is available. Add other file as you see fit. Make sure that If the prediction and pipeline is reproducible even if I loss access to the claude code session.

"Other" folder. Containing .env file python .env folder and all other file that you created.

In the root directory, not in any folder, let there be only AGENTS.md

## Communication method

i have connected you to discord. Communicate to me via discord DM to username:yodharin and I will reply.
For Example, if you need me to point to a direction. I will reply the DM or type the prompt.
DM me if you make progress. Please check DM continuously.

## Try the following for better score

if the competition rule allow,use external data and pretrain model.

Read the discussion forum in the competition website for help.

Search the internet for help.

## Daily task

1. Everyday, once per day when the cpu load is low, backup the project to github. Use a deployment key that you created.
2. From the data that you have, DM the the estimated probability of getting a prize with the current budget and with more budget that you can requested.
3. Fill in the table in the csv file in Z:competition.csv with Competition name, Current rank, End date,usd spent and your estimated probability of winning prizes.



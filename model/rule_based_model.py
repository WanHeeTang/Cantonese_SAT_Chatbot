import nltk

from model.models import UserModelSession, Choice, UserModelRun, Protocol
# For advance dataset
from model.classifiers_pre_compute import get_emotion, get_sentence_score
# from model.classifiers import get_emotion, get_sentence_score
import pandas as pd
import numpy as np
import random
from collections import deque
import re
import datetime
import time

nltk.download("wordnet")
from nltk.corpus import wordnet  # noqa


class ModelDecisionMaker:
    def __init__(self):

        self.neutral = pd.read_csv('/home/wanhee/Cantonese_SAT_Chatbot/Dataset/Empathy Classification/Additional Dataset/Updated_neutral.csv', encoding='utf-8')
        self.old_man = pd.read_csv('/home/wanhee/Cantonese_SAT_Chatbot/Dataset/Empathy Classification/Additional Dataset/updated_old_man.csv', encoding='utf-8')
        self.old_lady = pd.read_csv('/home/wanhee/Cantonese_SAT_Chatbot/Dataset/Empathy Classification/Additional Dataset/updated_old_lady.csv', encoding='utf-8')
        self.young_man = pd.read_csv('/home/wanhee/Cantonese_SAT_Chatbot/Dataset/Empathy Classification/Additional Dataset/updated_young_man.csv', encoding='utf-8')
        self.young_lady = pd.read_csv('/home/wanhee/Cantonese_SAT_Chatbot/Dataset/Empathy Classification/Additional Dataset/updated_young_lady.csv', encoding='utf-8')

        # Titles from workshops (Title 7 adapted to give more information)
        self.PROTOCOL_TITLES = [
            "0: 沒有",
            "1: 和童年的自己建立關係",
            "2: 對著兩張自己的照片笑",
            "3: 和小時候的自己建立愛",
            "4: 承諾關心小時候的自己",
            "5: 和小時候的自己保持愛的關係",
            "6: 處理小時候痛苦的回憶",
            "7: 對生活產生熱情",
            "8: 放鬆面部同身體肌肉",
            "9: 欣賞大自然環境",
            "10: 認識自己小時候不良行爲",
            "11: 認識並且控制自己過度自戀或自責的行爲",   
            "12: 欣賞自己所有嘅成就",
            "13: 透過自我安慰來面對負面情緒",
            "14: 用笑來減壓",
            "15: 改變個人觀點來處理負面情緒", 
            "16: 强化自己内心的狀態",
            "17: 解決個人危機",
            "18: 無傷害性地嘲笑㒼塞(音: mang4 sak1) 的想法", 
            "19: 通過體驗不同思考框架來提升想象力",
            "20: 利用名言來鼓勵自己",
        ]

        self.TITLE_TO_PROTOCOL = {
            self.PROTOCOL_TITLES[i]: i for i in range(len(self.PROTOCOL_TITLES))
        }

        self.recent_protocols = deque(maxlen=20)
        self.reordered_protocol_questions = {}
        self.protocols_to_suggest = []

        # Goes from user id to actual value
        self.current_run_ids = {}
        self.current_protocol_ids = {}

        self.current_protocols = {}

        self.positive_protocols = [i for i in range(1, 21)]

        self.INTERNAL_PERSECUTOR_PROTOCOLS = [
            self.PROTOCOL_TITLES[8],
            self.PROTOCOL_TITLES[13],
            self.PROTOCOL_TITLES[16],
            self.PROTOCOL_TITLES[19],
        ]

        # Keys: user ids, values: dictionaries describing each choice (in list)
        # and current choice
        self.user_choices = {}

        # Keys: user ids, values: scores for each question
        #self.user_scores = {}

        # Keys: user ids, values: current suggested protocols
        self.suggestions = {}

        # Tracks current emotion of each user after they classify it
        self.user_emotions = {}

        self.guess_emotion_predictions = {}
        # Structure of dictionary: {question: {
        #                           model_prompt: str or list[str],
        #                           choices: {maps user response to next protocol},
        #                           protocols: {maps user response to protocols to suggest},
        #                           }, ...
        #                           }
        # This could be adapted to be part of a JSON file (would need to address
        # mapping callable functions over for parsing).

        self.users_names = {}
        self.remaining_choices = {}

        self.recent_questions = {}

        self.chosen_personas = {}
        self.datasets = {}


        self.QUESTIONS = {

            "ask_name": {
               "model_prompt": "請問你想我點稱呼你？",
               "choices": {
                   "open_text": lambda user_id, db_session, curr_session, app: self.save_name(user_id)
               },
               "protocols": {"open_text": []},
           },


           "choose_persona": {
              "model_prompt": "請問你想同邊個傾計?",
              "choices": {
                  "信曦": lambda user_id, db_session, curr_session, app: self.get_neutral(user_id),
                  "權叔": lambda user_id, db_session, curr_session, app: self.get_old_man(user_id),
                  "霞姨": lambda user_id, db_session, curr_session, app: self.get_old_lady(user_id),
                  "偉文": lambda user_id, db_session, curr_session, app: self.get_young_man(user_id),
                  "子娟": lambda user_id, db_session, curr_session, app: self.get_young_lady(user_id),
              },
              "protocols": {
                  "信曦": [],
                  "權叔": [],
                  "霞姨": [],
                  "偉文": [],
                  "子娟": [],
              },
          },


            "opening_prompt": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_opening_prompt(user_id),

                "choices": {
                    "open_text": lambda user_id, db_session, curr_session, app: self.determine_next_prompt_opening(user_id, app, db_session)
                },
                "protocols": {"open_text": []},
            },

            "guess_emotion": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_guess_emotion(
                    user_id, app, db_session
                ),
                "choices": {
                    "是": {
                        "唔開心": "after_classification_sad",
                        "嬲": "after_classification_angry",
                        "擔心": "after_classification_fear",
                        "開心": "after_classification_happy",
                    },
                    "否": "check_emotion",
                },
                "protocols": {
                    "是": [],
                    "否": []
                    },
            },


            "check_emotion": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_check_emotion(user_id, app, db_session),

                "choices": {
                    "唔開心": lambda user_id, db_session, curr_session, app: self.get_sad_emotion(user_id),
                    "嬲": lambda user_id, db_session, curr_session, app: self.get_angry_emotion(user_id),
                    "擔心": lambda user_id, db_session, curr_session, app: self.get_anxious_emotion(user_id),
                    "開心": lambda user_id, db_session, curr_session, app: self.get_happy_emotion(user_id),
                },
                "protocols": {
                    "唔開心": [],
                    "嬲": [],
                    "擔心" : [],
                    "開心": []
                },
            },

            ############ HAPPY EMOTIONS #############

            "after_classification_happy": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_happy(user_id, app, db_session),

                "choices": {
                    "好啊": "suggestions",
                    "不用了": "ending_prompt"
                },
                "protocols": {
                    "好啊": [self.PROTOCOL_TITLES[9], self.PROTOCOL_TITLES[12], self.PROTOCOL_TITLES[13]], #change here?
                    #[self.PROTOCOL_TITLES[k] for k in self.positive_protocols],
                    "不用了": []
                },
            },

            ############# SAD EMOTIONS ################
            "after_classification_sad": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_specific_event(user_id, app, db_session),

                "choices": {
                    "對，有特定事情導致": "event_is_recent",
                    "沒有，只是一種感覺": "more_questions",
                },
                "protocols": {
                    "對，有特定事情導致": [],
                    "沒有，只是一種感覺": []
                },
            },

            "event_is_recent": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_event_is_recent(user_id, app, db_session),

                "choices": {
                    "最近發生": "revisiting_recent_events",
                    "好耐之前": "revisiting_distant_events",
                },
                "protocols": {
                    "最近發生": [],
                    "好耐之前": []
                    },
            },

            "revisiting_recent_events": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_revisit_recent(user_id, app, db_session),

                "choices": {
                    "有": "more_questions",
                    "沒有": "more_questions",
                },
                "protocols": {
                    "有": [self.PROTOCOL_TITLES[7], self.PROTOCOL_TITLES[8]],
                    "沒有": [self.PROTOCOL_TITLES[13]],
                },
            },

            "revisiting_distant_events": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_revisit_distant(user_id, app, db_session),

                "choices": {
                    "有": "more_questions",
                    "沒有": "more_questions",
                },
                "protocols": {
                    "有": [self.PROTOCOL_TITLES[15], self.PROTOCOL_TITLES[17]],
                    "沒有": [self.PROTOCOL_TITLES[6]]
                },
            },

            "more_questions": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_more_questions(user_id, app, db_session),

                "choices": {
                    "好": lambda user_id, db_session, curr_session, app: self.get_next_question(user_id),
                    "唔好": "project_emotion",
                },
                "protocols": {
                    "好": [],
                    "唔好": [self.PROTOCOL_TITLES[15]],
                },
            },

            "displaying_antisocial_behaviour": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_antisocial(user_id, app, db_session),

                "choices": {
                    "有": "project_emotion",
                    "沒有": lambda user_id, db_session, curr_session, app: self.get_next_question(user_id),
                },
                "protocols": {
                    "有": [self.PROTOCOL_TITLES[15], self.PROTOCOL_TITLES[10]],
                    "沒有": [self.PROTOCOL_TITLES[15]],
                },
            },

            "internal_persecutor_saviour": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_saviour(user_id, app, db_session),

                "choices": {
                    "會": "project_emotion",
                    "唔會": "internal_persecutor_victim",
                },
                "protocols": {
                    "會": self.INTERNAL_PERSECUTOR_PROTOCOLS,
                    "唔會": []
                },
            },

            "internal_persecutor_victim": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_victim(user_id, app, db_session),

                "choices": {
                    "是": "project_emotion",
                    "不是": "internal_persecutor_controlling",
                },
                "protocols": {
                    "是": self.INTERNAL_PERSECUTOR_PROTOCOLS,
                    "不是": []
                },
            },

            "internal_persecutor_controlling": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_controlling(user_id, app, db_session),

                "choices": {
                "有": "project_emotion",
                "沒有": "internal_persecutor_accusing"
                },
                "protocols": {
                "有": self.INTERNAL_PERSECUTOR_PROTOCOLS,
                "沒有": []
                },
            },

            "internal_persecutor_accusing": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_accusing(user_id, app, db_session),

                "choices": {
                "有": "project_emotion",
                "沒有": lambda user_id, db_session, curr_session, app: self.get_next_question(user_id),
                },
                "protocols": {
                "有": self.INTERNAL_PERSECUTOR_PROTOCOLS,
                "沒有": [self.PROTOCOL_TITLES[15]],
                },
            },

            "rigid_thought": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_rigid_thought(user_id, app, db_session),

                "choices": {
                    "有": lambda user_id, db_session, curr_session, app: self.get_next_question(user_id),
                    "沒有": "project_emotion",
                },
                "protocols": {
                    "有": [self.PROTOCOL_TITLES[15]],
                    "沒有": [self.PROTOCOL_TITLES[15], self.PROTOCOL_TITLES[19]],
                },
            },


            "personal_crisis": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_personal_crisis(user_id, app, db_session),

                "choices": {
                    "有": "project_emotion",
                    "沒有": lambda user_id, db_session, curr_session, app: self.get_next_question(user_id),
                },
                "protocols": {
                    "有": [self.PROTOCOL_TITLES[15], self.PROTOCOL_TITLES[17]],
                    "沒有": [self.PROTOCOL_TITLES[15]],
                },
            },

            ############# ANGRY EMOTION ###########
            "after_classification_angry": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_specific_event(user_id, app, db_session),

                "choices": {
                    "對，有特定事情導致": "event_is_recent",
                    "沒有，只是一種感覺": "more_questions",
                },
                "protocols": {
                    "對，有特定事情導致": [],
                    "沒有，只是一種感覺": []
                },
            },

            ############# FEAR/ANXIOUS EMOTIONS ###########

            "after_classification_fear": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_specific_event(user_id, app, db_session),

                "choices": {
                    "對，有特定事情導致": "event_is_recent",
                    "沒有，只是一種感覺": "more_questions",
                },
                "protocols": {
                    "對，有特定事情導致": [],
                    "沒有，只是一種感覺": []
                },
            },

            ############################# ALL EMOTIONS #############################

            "project_emotion": {
               "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_project_emotion(user_id, app, db_session),

               "choices": {
                   "繼續": "suggestions",
               },
               "protocols": {
                   "繼續": [],
               },
            },

            "suggestions": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_suggestions(user_id, app, db_session),

                "choices": {
                     self.PROTOCOL_TITLES[k]: "trying_protocol" #self.current_protocol_ids[user_id]
                     for k in self.positive_protocols
                },
                "protocols": {
                     self.PROTOCOL_TITLES[k]: [self.PROTOCOL_TITLES[k]]
                     for k in self.positive_protocols
                },
            },

            "trying_protocol": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_trying_protocol(user_id, app, db_session),

                "choices": {
                    "繼續": "user_found_useful"
                },
                "protocols": {
                    "繼續": []
                },
            },

            "user_found_useful": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_found_useful(user_id, app, db_session),

                "choices": {
                    "我覺得好咗": "new_protocol_better",
                    "我覺得差咗": "new_protocol_worse",
                    "我覺得冇變": "new_protocol_same",
                },
                "protocols": {
                    "我覺得好咗": [],
                    "我覺得差咗": [],
                    "我覺得冇變": []
                },
            },

            "new_protocol_better": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_new_better(user_id, app, db_session),

                "choices": {
                    "好啊（繼續其他練習）": lambda user_id, db_session, curr_session, app: self.determine_next_prompt_new_protocol(
                        user_id, app
                    ),
                    "好啊（重新開始）": "restart_prompt",
                    "不用了（結束）": "ending_prompt",
                },
                "protocols": {
                    "好啊（繼續其他練習）": [],
                    "好啊（重新開始）": [],
                    "不用了（結束）": []
                },
            },

            "new_protocol_worse": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_new_worse(user_id, app, db_session),

                "choices": {
                    "好啊（繼續其他練習）": lambda user_id, db_session, curr_session, app: self.determine_next_prompt_new_protocol(
                        user_id, app
                    ),
                    "好啊（重新開始）": "restart_prompt",
                    "不用了（結束）": "ending_prompt",
                },
                "protocols": {
                    "好啊（繼續其他練習）": [],
                    "好啊（重新開始）": [],
                    "不用了（結束）": []
                },
            },

            "new_protocol_same": {
                "model_prompt": [
                                "呢種情況有時都會發生，如果你想，我可以建議其他練習，或者會對你更有幫助。",
                                "你想我建議其他練習嗎？"
                                ],

                "choices": {
                    "好啊（繼續其他練習）": lambda user_id, db_session, curr_session, app: self.determine_next_prompt_new_protocol(
                        user_id, app
                    ),
                    "好啊（重新來過）": "restart_prompt",
                    "不用了（結束）": "ending_prompt",  
                },
                "protocols": {
                    "好啊（繼續其他練習）": [],
                    "好啊（重新來過）": [],
                    "不用了（結束）": []
                },
            },

            "ending_prompt": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_model_prompt_ending(user_id, app, db_session),

                "choices": {"any": "opening_prompt"},
                "protocols": {"any": []}
                },

            "restart_prompt": {
                "model_prompt": lambda user_id, db_session, curr_session, app: self.get_restart_prompt(user_id),

                "choices": {
                    "open_text": lambda user_id, db_session, curr_session, app: self.determine_next_prompt_restart(user_id, app, db_session)
                },
                "protocols": {"open_text": []},
            },
        }
        self.QUESTION_KEYS = list(self.QUESTIONS.keys())

    def initialise_prev_questions(self, user_id):
        self.recent_questions[user_id] = []

    def clear_persona(self, user_id):
        self.chosen_personas[user_id] = ""

    def clear_names(self, user_id):
        self.users_names[user_id] = ""

    def clear_datasets(self, user_id):
        self.datasets[user_id] = pd.DataFrame(columns=['sentences'])

    def initialise_remaining_choices(self, user_id):
        self.remaining_choices[user_id] = ["displaying_antisocial_behaviour", "internal_persecutor_saviour", "personal_crisis", "rigid_thought"]


    def save_name(self, user_id):
        try:
            user_response = self.user_choices[user_id]["choices_made"]["ask_name"]
        except:  # noqa
            user_response = ""
        self.users_names[user_id] = user_response
        return "choose_persona"


    def get_suggestions(self, user_id, app): #from all the lists of protocols collected at each step of the dialogue it puts together some and returns these as suggestions
        suggestions = []
        for curr_suggestions in list(self.suggestions[user_id]):
            if len(curr_suggestions) > 2:
                i, j = random.choices(range(0,len(curr_suggestions)), k=2)
                if curr_suggestions[i] and curr_suggestions[j] in self.PROTOCOL_TITLES: #weeds out some gibberish that im not sure why it's there
                    suggestions.extend([curr_suggestions[i], curr_suggestions[j]])
            else:
                suggestions.extend(curr_suggestions)
            suggestions = set(suggestions)
            suggestions = list(suggestions)
        while len(suggestions) < 4: #augment the suggestions if less than 4, we add random ones avoiding repetitions
            p = random.choice([i for i in range(1,20) if i not in [6,13]]) #Update! we dont want to suggest protocol 6 or 13 at random here
            if (any(self.PROTOCOL_TITLES[p] not in curr_suggestions for curr_suggestions in list(self.suggestions[user_id]))
                and self.PROTOCOL_TITLES[p] not in self.recent_protocols and self.PROTOCOL_TITLES[p] not in suggestions):
                        suggestions.append(self.PROTOCOL_TITLES[p])
                        self.suggestions[user_id].extend([self.PROTOCOL_TITLES[p]])
        

        suggestions = [i.split(":") for i in suggestions]
        lst = []

        suggestions=sorted(suggestions,key=lambda x:int(x[0]))

        for i, v in enumerate(suggestions):
            lst.append(":".join(v))

        return lst


    def clear_suggestions(self, user_id):
        self.suggestions[user_id] = []
        self.reordered_protocol_questions[user_id] = deque(maxlen=5)

    def clear_emotion_scores(self, user_id):
        self.guess_emotion_predictions[user_id] = ""

    def create_new_run(self, user_id, db_session, user_session):
        new_run = UserModelRun(session_id=user_session.id)
        db_session.add(new_run)
        db_session.commit()
        self.current_run_ids[user_id] = new_run.id
        return new_run

    def clear_choices(self, user_id):
        self.user_choices[user_id] = {}

    def update_suggestions(self, user_id, protocols, app):

        # Check if user_id already has suggestions
        try:
            self.suggestions[user_id]
        except KeyError:
            self.suggestions[user_id] = []

        if type(protocols) != list:
            self.suggestions[user_id].append(deque([protocols]))
        else:
            self.suggestions[user_id].append(deque(protocols))

    # Takes next item in queue, or moves on to suggestions
    # if all have been checked

    def get_neutral(self, user_id):
       self.chosen_personas[user_id] = "信曦"
       self.datasets[user_id] = self.neutral
       return "opening_prompt"
    def get_old_man(self, user_id):
       self.chosen_personas[user_id] = "權叔"
       self.datasets[user_id] = self.old_man
       return "opening_prompt"
    def get_old_lady(self, user_id):
       self.chosen_personas[user_id] = "霞姨"
       self.datasets[user_id] = self.old_lady
       return "opening_prompt"
    def get_young_man(self, user_id):
       self.chosen_personas[user_id] = "偉文"
       self.datasets[user_id] = self.young_man
       return "opening_prompt"
    def get_young_lady(self, user_id):
       self.chosen_personas[user_id] = "子娟"
       self.datasets[user_id] = self.young_lady
       return "opening_prompt"


    def get_opening_prompt(self, user_id):
        # time.sleep(7)
        if self.users_names[user_id] == "":
            opening_prompt = ["你好！ 我是 " + self.chosen_personas[user_id] + "！ ", "你今日覺得點啊？"]
        else:
            opening_prompt = ["Hello " + self.users_names[user_id] + "! 我係" + self.chosen_personas[user_id] + "。 ", "你今日覺得點啊？"]
        return opening_prompt


    def get_restart_prompt(self, user_id):
        # time.sleep(7)
        if self.users_names[user_id] == "":
            restart_prompt = ["請你再講多次，你今日覺得點啊？"]
        else:
            restart_prompt = ["請你再講多次， " + self.users_names[user_id] + "， 你今日覺得點啊？"]
        return restart_prompt

    def get_next_question(self, user_id):
        if self.remaining_choices[user_id] == []:
            return "project_emotion"
        else:
            selected_choice = np.random.choice(self.remaining_choices[user_id])
            self.remaining_choices[user_id].remove(selected_choice)
            return selected_choice

    def add_to_reordered_protocols(self, user_id, next_protocol):
        self.reordered_protocol_questions[user_id].append(next_protocol)

    def add_to_next_protocols(self, next_protocols):
        self.protocols_to_suggest.append(deque(next_protocols))

    def clear_suggested_protocols(self):
        self.protocols_to_suggest = []

    # NOTE: this is not currently used, but can be integrated to support
    # positive protocol suggestions (to avoid recent protocols).
    # You would need to add it in when a user's emotion is positive
    # and they have chosen a protocol.

    def add_to_recent_protocols(self, recent_protocol):
        if len(self.recent_protocols) == self.recent_protocols.maxlen:
            # Removes oldest protocol
            self.recent_protocols.popleft()
        self.recent_protocols.append(recent_protocol)


    def determine_next_prompt_opening(self, user_id, app, db_session):
        user_response = self.user_choices[user_id]["choices_made"]["opening_prompt"]
        emotion = get_emotion(user_response)
        #emotion = np.random.choice(["Happy", "Sad", "Angry", "Anxious"]) #random choice to be replaced with emotion classifier
        if emotion == "擔心":
            self.guess_emotion_predictions[user_id] = "擔心"
            self.user_emotions[user_id] = "擔心"
        elif emotion == "唔開心":
            self.guess_emotion_predictions[user_id] = "唔開心"
            self.user_emotions[user_id] = "唔開心"
        elif emotion == "嬲":
            self.guess_emotion_predictions[user_id] = "嬲"
            self.user_emotions[user_id] = "嬲"
        else:
            self.guess_emotion_predictions[user_id] = "開心"
            self.user_emotions[user_id] ="開心"
        #self.guess_emotion_predictions[user_id] = emotion
        #self.user_emotions[user_id] = emotion
        return "guess_emotion"

    def determine_next_prompt_restart(self, user_id, app, db_session):
        self.clear_suggestions(user_id)
        self.remaining_choices[user_id] = ["displaying_antisocial_behaviour", "internal_persecutor_saviour", "personal_crisis", "rigid_thought"]
        user_response = self.user_choices[user_id]["choices_made"]["restart_prompt"]
        emotion = get_emotion(user_response)
        #emotion = np.random.choice(["Happy", "Sad", "Angry", "Anxious"]) #random choice to be replaced with emotion classifier
        if emotion == "擔心":
            self.guess_emotion_predictions[user_id] = "擔心"
            self.user_emotions[user_id] = "擔心"
        elif emotion == "唔開心":
            self.guess_emotion_predictions[user_id] = "唔開心"
            self.user_emotions[user_id] = "唔開心"
        elif emotion == "嬲":
            self.guess_emotion_predictions[user_id] = "嬲"
            self.user_emotions[user_id] = "嬲"
        else:
            self.guess_emotion_predictions[user_id] = "開心"
            self.user_emotions[user_id] ="開心"
        #self.guess_emotion_predictions[user_id] = emotion
        #self.user_emotions[user_id] = emotion
        return "guess_emotion"


    def get_best_sentence(self, column, prev_qs):
        #return random.choice(column.dropna().sample(n=15).to_list()) #using random choice instead of machine learning
        maxscore = 0
        chosen = ''
        # for row in column.dropna().sample(n=5): #was 25
        for row in column.dropna():
            # print(row)
            fitscore = get_sentence_score(row, prev_qs)
            if fitscore > maxscore:
                maxscore = fitscore
                chosen = row
        if chosen != '':
            return chosen
        else:
            return random.choice(column.dropna().sample(n=5).to_list()) #was 25


    def split_sentence(self, sentence):
        temp_list = re.split('（？《=【。！】）+(?<=[.?!]) +', sentence)
        if '' in temp_list:
            temp_list.remove('')
        temp_list = [i + " " if i[-1] in ["。", "？", "！"] else i for i in temp_list]
        if len(temp_list) == 2:
            return temp_list[0], temp_list[1]
        elif len(temp_list) == 3:
            return temp_list[0], temp_list[1], temp_list[2]
        else:
            return sentence


    def get_model_prompt_guess_emotion(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        column = data["所有情緒- 根據你所說的，我相信你正在感受{}。這個對嗎？"].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        question = my_string.format(self.guess_emotion_predictions[user_id].lower())
        return self.split_sentence(question)

    def get_model_prompt_check_emotion(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        column = data["所有的情緒- 對不起。請從以下最能反映你的感受的情緒中選擇："].dropna()
        my_string = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(my_string)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(my_string)
        return self.split_sentence(my_string)

    def get_sad_emotion(self, user_id):
        self.guess_emotion_predictions[user_id] = "唔開心"
        self.user_emotions[user_id] = "唔開心"
        return "after_classification_sad"
    def get_angry_emotion(self, user_id):
        self.guess_emotion_predictions[user_id] = "嬲"
        self.user_emotions[user_id] = "嬲"
        return "after_classification_angry"
    def get_anxious_emotion(self, user_id):
        self.guess_emotion_predictions[user_id] = "擔心"
        self.user_emotions[user_id] = "擔心"
        return "after_classification_fear"
    def get_happy_emotion(self, user_id):
        self.guess_emotion_predictions[user_id] = "開心"
        self.user_emotions[user_id] = "開心"
        return "after_classification_happy"

    def get_model_prompt_project_emotion(self, user_id, app, db_session):
        # time.sleep(7)
        if self.chosen_personas[user_id] == "權叔":
            prompt = "好的，謝謝！而家到左最後一樣最重要嘅事情，既然你認爲你係 ‘" + self.user_emotions[user_id].lower() + "’ 我希望你可以嘗試將呢種情緒投射到你童年時候嘅自己身上。當你準備好嘅時候就可以㩒 ‘繼續’，然後我會建議一D我認爲適合你嘅練習。"
        elif self.chosen_personas[user_id] == "霞姨":
            prompt = "謝謝！我將會推薦一D練習俾你。在我這樣做以先，你可以試下將你 ‘" + self.user_emotions[user_id].lower() + "’ 嘅感覺投射到你童年時候嘅自己身上嗎？請慢慢嘗試，當你可以嘅時候就可以㩒 ‘繼續’。"
        elif self.chosen_personas[user_id] == "偉文":
            prompt = "好的，感謝你俾我知道。係我建議一D練習給你之前，請先用D時間將你而家 ‘" + self.user_emotions[user_id].lower() + "’ 嘅感覺投射到你童年時候嘅自己身上。當你覺得你可以嘅時候就可以㩒 ‘繼續’。"
        else:
            prompt = "感謝你！係我諗緊邊個練習最適合你嘅時候，你可以慢慢咁將你而家 ‘" + self.user_emotions[user_id].lower() + "’ 嘅情緒咁投射到你童年時候嘅自己身上。當你完成嘅時候，請㩒 ‘繼續’，然後我就會同你講我建議嘅練習。"
        return self.split_sentence(prompt)


    def get_model_prompt_saviour(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = self.user_emotions[user_id] + "——你認為你應該成為別人的救世主嗎？"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_victim(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = self.user_emotions[user_id] + "——你是否認為自己是受害者，將你的負面情緒歸咎於別人？"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_controlling(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = self.user_emotions[user_id] + "——你覺得你在試圖控制某人嗎？"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_accusing(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = self.user_emotions[user_id] + "- 當出現問題時，你是否總是責備和指責自己？"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_specific_event(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = self.user_emotions[user_id] + "- 這是由特定事件引起的嗎？"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_event_is_recent(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = self.user_emotions[user_id] + "- 這是由最近或遙遠的事件（或事件）引起的嗎？"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_revisit_recent(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = self.user_emotions[user_id] + "- 你最近是否嘗試過練習13，並發現由於舊事件而重新點燃了無法控制的情緒？"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_revisit_distant(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = self.user_emotions[user_id] + " - 你最近是否嘗試過練習6，並發現由於舊事件而重新點燃了無法控制的情緒？"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_more_questions(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = self.user_emotions[user_id] + "- 謝謝。現在我會問一些問題來了解你的情況。"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_antisocial(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = self.user_emotions[user_id] + "- 你是否對某人有強烈的感受或表達過以下任何情緒："
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return [self.split_sentence(question), "嫉妒，妒忌，仇恨，懷疑/不信任？"]

    def get_model_prompt_rigid_thought(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = self.user_emotions[user_id] + "- 在之前的對話中，你是否考慮過其他觀點？"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_personal_crisis(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        base_prompt = self.user_emotions[user_id] + "- 你是否正在經歷個人危機（與親人相處時遇到困難，例如與朋友吵架）？"
        column = data[base_prompt].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_happy(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        column = data["快樂- 這很好！讓我推薦一個你可以嘗試的練習。"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_suggestions(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        column = data["所有情緒- 這是我的建議，請選擇你想嘗試的練習"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_trying_protocol(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        column = data["所有情緒- 請現在嘗試通過此練習。完成後，按“繼續”"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return ["你揀咗練習 " + str(self.current_protocol_ids[user_id][0]) + "。 ", self.split_sentence(question)]

    def get_model_prompt_found_useful(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        column = data["所有情緒- 服用此練習後，你感覺好些還是壞些？"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_new_better(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        column = data["所有的情緒- 你想嘗試另一個練習嗎？ （病人感覺好多了）"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_new_worse(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        column = data["所有的情緒- 你想嘗試另一個練習嗎？ （患者感覺更糟）"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return self.split_sentence(question)

    def get_model_prompt_ending(self, user_id, app, db_session):
        prev_qs = pd.DataFrame(self.recent_questions[user_id],columns=['sentences'])
        data = self.datasets[user_id]
        column = data["所有情緒- 感謝你的參與。再見"].dropna()
        question = self.get_best_sentence(column, prev_qs)
        if len(self.recent_questions[user_id]) < 50:
            self.recent_questions[user_id].append(question)
        else:
            self.recent_questions[user_id] = []
            self.recent_questions[user_id].append(question)
        return [self.split_sentence(question), "你已斷線, 請更新頁面來重新連線。"]


    def determine_next_prompt_new_protocol(self, user_id, app):
        try:
            self.suggestions[user_id]
        except KeyError:
            self.suggestions[user_id] = []
        if len(self.suggestions[user_id]) > 0:
            return "suggestions"
        return "more_questions"


    def determine_positive_protocols(self, user_id, app):
        protocol_counts = {}
        total_count = 0

        for protocol in self.positive_protocols:
            count = Protocol.query.filter_by(protocol_chosen=protocol).count()
            protocol_counts[protocol] = count
            total_count += count

        # for protocol in counts:
        if total_count > 10:
            first_item = min(zip(protocol_counts.values(), protocol_counts.keys()))[1]
            del protocol_counts[first_item]

            second_item = min(zip(protocol_counts.values(), protocol_counts.keys()))[1]
            del protocol_counts[second_item]

            third_item = min(zip(protocol_counts.values(), protocol_counts.keys()))[1]
            del protocol_counts[third_item]
        else:
            # CASE: < 10 protocols undertaken in total, so randomness introduced
            # to avoid lowest 3 being recommended repeatedly.
            # Gives number of next protocol to be suggested
            first_item = np.random.choice(
                list(set(self.positive_protocols) - set(self.recent_protocols))
            )
            second_item = np.random.choice(
                list(
                    set(self.positive_protocols)
                    - set(self.recent_protocols)
                    - set([first_item])
                )
            )
            third_item = np.random.choice(
                list(
                    set(self.positive_protocols)
                    - set(self.recent_protocols)
                    - set([first_item, second_item])
                )
            )

        return [
            self.PROTOCOL_TITLES[first_item],
            self.PROTOCOL_TITLES[second_item],
            self.PROTOCOL_TITLES[third_item],
        ]

    def determine_protocols_keyword_classifiers(
        self, user_id, db_session, curr_session, app
    ):

        # We add "suggestions" first, and in the event there are any left over we use those, otherwise we divert past it.
        self.add_to_reordered_protocols(user_id, "suggestions")

        # Default case: user should review protocols 13 and 14.
        #self.add_to_next_protocols([self.PROTOCOL_TITLES[13], self.PROTOCOL_TITLES[14]])
        return self.get_next_protocol_question(user_id, app)


    def update_conversation(self, user_id, new_dialogue, db_session, app):
        try:
            session_id = self.user_choices[user_id]["current_session_id"]
            curr_session = UserModelSession.query.filter_by(id=session_id).first()
            if curr_session.conversation is None:
                curr_session.conversation = "" + new_dialogue
            else:
                curr_session.conversation = curr_session.conversation + new_dialogue
            curr_session.last_updated = datetime.datetime.utcnow()
            db_session.commit()
        except KeyError:
            curr_session = UserModelSession(
                user_id=user_id,
                conversation=new_dialogue,
                last_updated=datetime.datetime.utcnow(),
            )

            db_session.add(curr_session)
            db_session.commit()
            self.user_choices[user_id]["current_session_id"] = curr_session.id


    def save_current_choice(
        self, user_id, input_type, user_choice, user_session, db_session, app
    ):
        # Set up dictionary if not set up already
        # with Session() as session:

        try:
            self.user_choices[user_id]
        except KeyError:
            self.user_choices[user_id] = {}

        # Define default choice if not already set
        try:
            current_choice = self.user_choices[user_id]["choices_made"][
                "current_choice"
            ]
        except KeyError:
            current_choice = self.QUESTION_KEYS[0]

        try:
            self.user_choices[user_id]["choices_made"]
        except KeyError:
            self.user_choices[user_id]["choices_made"] = {}

        if current_choice == "ask_name":
            self.clear_suggestions(user_id)
            self.user_choices[user_id]["choices_made"] = {}
            self.create_new_run(user_id, db_session, user_session)

        # Save current choice
        self.user_choices[user_id]["choices_made"]["current_choice"] = current_choice
        self.user_choices[user_id]["choices_made"][current_choice] = user_choice

        curr_prompt = self.QUESTIONS[current_choice]["model_prompt"]
        # prompt_to_use = curr_prompt
        if callable(curr_prompt):
            curr_prompt = curr_prompt(user_id, db_session, user_session, app)

        #removed stuff here

        else:
            self.update_conversation(
                user_id,
                "Model:{} \nUser:{} \n".format(curr_prompt, user_choice),
                db_session,
                app,
            )

        # Case: update suggestions for next attempt by removing relevant one
        if (
            current_choice == "suggestions"
        ):

            # PRE: user_choice is a string representing a number from 1-20,
            # or the title for the corresponding protocol

            try:
                current_protocol = self.TITLE_TO_PROTOCOL[user_choice]
            except KeyError:
                current_protocol = int(user_choice)

            protocol_chosen = Protocol(
                protocol_chosen=current_protocol,
                user_id=user_id,
                session_id=user_session.id,
                run_id=self.current_run_ids[user_id],
            )
            db_session.add(protocol_chosen)
            db_session.commit()
            self.current_protocol_ids[user_id] = [current_protocol, protocol_chosen.id]

            for i in range(len(self.suggestions[user_id])):
                curr_protocols = self.suggestions[user_id][i]
                if curr_protocols[0] == self.PROTOCOL_TITLES[current_protocol]:
                    curr_protocols.popleft()
                    if len(curr_protocols) == 0:
                        self.suggestions[user_id].pop(i)
                    break

        # PRE: User choice is string in ["Better", "Worse"]
        elif current_choice == "user_found_useful":
            current_protocol = Protocol.query.filter_by(
                id=self.current_protocol_ids[user_id][1]
            ).first()
            current_protocol.protocol_was_useful = user_choice
            db_session.commit()

        if current_choice == "guess_emotion":
            option_chosen = user_choice + " ({})".format(
                self.guess_emotion_predictions[user_id]
            )
        else:
            option_chosen = user_choice
        choice_made = Choice(
            choice_desc=current_choice,
            option_chosen=option_chosen,
            user_id=user_id,
            session_id=user_session.id,
            run_id=self.current_run_ids[user_id],
        )
        db_session.add(choice_made)
        db_session.commit()

        return choice_made

    def determine_next_choice(
        self, user_id, input_type, user_choice, db_session, user_session, app
    ):
        # Find relevant user info by using user_id as key in dict.
        #
        # Then using the current choice and user input, we determine what the next
        # choice is and return this as the output.

        # Some edge cases to consider based on the different types of each field:
        # May need to return list of model responses. For next protocol, may need
        # to call function if callable.

        # If we cannot find the specific choice (or if None etc.) can set user_choice
        # to "any".

        # PRE: Will be defined by save_current_choice if it did not already exist.
        # (so cannot be None)

        current_choice = self.user_choices[user_id]["choices_made"]["current_choice"]
        current_choice_for_question = self.QUESTIONS[current_choice]["choices"]
        current_protocols = self.QUESTIONS[current_choice]["protocols"]
        if input_type != "open_text":
            if (
                current_choice != "suggestions"
                and current_choice != "event_is_recent"
                and current_choice != "more_questions"
                and current_choice != "user_found_useful"
                and current_choice != "check_emotion"
                and current_choice != "new_protocol_better"
                and current_choice != "new_protocol_worse"
                and current_choice != "new_protocol_same"
                and current_choice != "choose_persona"
                and current_choice != "project_emotion"
                and current_choice != "after_classification_happy"
                and current_choice != "after_classification_sad"
                and current_choice != "after_classification_angry"
                and current_choice != "after_classification_fear"
            ):
                user_choice = user_choice.lower()

            if (
                current_choice == "suggestions"
            ):
                try:
                    current_protocol = self.TITLE_TO_PROTOCOL[user_choice]
                except KeyError:
                    # User can input choice
                    current_protocol = int(user_choice)
                protocol_choice = self.PROTOCOL_TITLES[current_protocol]
                next_choice = current_choice_for_question[protocol_choice]
                protocols_chosen = current_protocols[protocol_choice]

            elif current_choice == "check_emotion":
                if user_choice == "唔開心":
                    next_choice = current_choice_for_question["唔開心"]
                    protocols_chosen = current_protocols["唔開心"]
                elif user_choice == "嬲":
                    next_choice = current_choice_for_question["嬲"]
                    protocols_chosen = current_protocols["嬲"]
                elif user_choice == "擔心":
                    next_choice = current_choice_for_question["擔心"]
                    protocols_chosen = current_protocols["擔心"]
                else:
                    next_choice = current_choice_for_question["開心"]
                    protocols_chosen = current_protocols["開心"]
            else:
                if current_choice == "check_emotion":
                    if user_choice == "唔開心":
                        next_choice = current_choice_for_question["唔開心"]
                        protocols_chosen = current_protocols["唔開心"]
                    elif user_choice == "嬲":
                        next_choice = current_choice_for_question["嬲"]
                        protocols_chosen = current_protocols["嬲"]
                    elif user_choice == "擔心":
                        next_choice = current_choice_for_question["擔心"]
                        protocols_chosen = current_protocols["擔心"]
                    else:
                        next_choice = current_choice_for_question["開心"]
                        protocols_chosen = current_protocols["開心"]
                else:
                    next_choice = current_choice_for_question[user_choice]
                    protocols_chosen = current_protocols[user_choice]

        else:
            next_choice = current_choice_for_question["open_text"]
            protocols_chosen = current_protocols["open_text"]

        if callable(next_choice):
            next_choice = next_choice(user_id, db_session, user_session, app)

        if current_choice == "guess_emotion" and user_choice.lower() == "是":
            if self.guess_emotion_predictions[user_id] == "唔開心":
                next_choice = next_choice["唔開心"]
            elif self.guess_emotion_predictions[user_id] == "嬲":
                next_choice = next_choice["嬲"]
            elif self.guess_emotion_predictions[user_id] == "擔心":
                next_choice = next_choice["擔心"]
            else:
                next_choice = next_choice["開心"]

        if callable(protocols_chosen):
            protocols_chosen = protocols_chosen(user_id, db_session, user_session, app)
        next_prompt = self.QUESTIONS[next_choice]["model_prompt"]
        if callable(next_prompt):
            next_prompt = next_prompt(user_id, db_session, user_session, app)
        if (
            len(protocols_chosen) > 0
            and current_choice != "suggestions"
        ):
            self.update_suggestions(user_id, protocols_chosen, app)

        # Case: new suggestions being created after first protocol attempted
        if next_choice == "opening_prompt":
            self.clear_suggestions(user_id)
            self.clear_emotion_scores(user_id)
            self.create_new_run(user_id, db_session, user_session)

        if next_choice == "suggestions":
            next_choices = self.get_suggestions(user_id, app)

        else:
            next_choices = list(self.QUESTIONS[next_choice]["choices"].keys())
        self.user_choices[user_id]["choices_made"]["current_choice"] = next_choice
        return {"model_prompt": next_prompt, "choices": next_choices}

import threading

RTC_STORE = {}
RTC_STORE_LOCK = threading.Lock()

GLOBAL_KEY = "aloha_policy_global_obs"
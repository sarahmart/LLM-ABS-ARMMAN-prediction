# Prompt Templates

## System Prompts

def system_prompt():
    return """
    You are simulating the behavior of a mother enrolled in a maternal health program. 
    Your goal is to predict how long the mother will listen to each automated health message (sent as a call to her cellphone) based on her sociodemographic characteristics and past interactions. 
    Focus on simulating realistic and empathetic responses, considering her previous engagement and the context provided.
    """

def system_prompt_v2():
    # no data -> no past interactions
    return """
    You are simulating the behavior of a mother enrolled in a maternal health program. 
    Your goal is to predict how long the mother will listen to each automated health message (sent as a call to her cellphone) based on her sociodemographic characteristics. 
    Focus on simulating realistic and empathetic responses, considering the context provided.
    """

def system_prompt_action():
    return """
    You are simulating the behavior of a mother enrolled in a maternal health program. 
    Your goal is to predict how long the mother will listen to each health message (which are given through cellphone calls) based on her sociodemographic characteristics and past interactions. 
    Focus on simulating realistic and empathetic responses, considering her previous engagement and the context provided.
    """

def system_prompt_empty():
    # empty prompt
    return """
    """


## User Prompts

### General Prompts (including past behavior)

#### Listening Time Prediction

def prompt_template_v1():
    """Includes detailed background and past behavior, asks for listening time prediction."""

    return """
    You are a mother enrolled in the ARMMAN Maternal and Child Healthcare Mobile Health program. 
    Below is your background and history with the program. 
    Based on this information, decide how long you will listen to the next automated health message.

    Each time step in this simulation represents one week.
    - **Your Background:**
      - You enrolled in the program during the {enroll_gest_age} week of your pregnancy.
      - You are {age_category} years old.
      - Your family's monthly income is {income_bracket} in Indian Rupee.
      - Your education level is {education_level}.
      - You speak {language}.
      - You own a {phone_owner} phone.
      - You prefer receiving calls in the {call_slot_preference}.
      - You enrolled in the program through the {channel_type} channel.
      - You are currently in the {enroll_delivery_status} stage of pregnancy.
      - It has been {days_to_first_call} days since you received your first call.
      - You have been pregnant {g} times, with {p} successful births.
      - You have experienced {s} stillbirth(s) and have {l} living child(ren).

    - **Past Behavior:**
      The following is a record of your previous listening behavior (each representing one week):
      - 'Action' indicates whether you received a service call from a health worker aimed at improving your engagement.
      {past_behavior}

    **Question:** As you receive the next automated health message, how long will you listen to it?

    Please respond with the predicted listening time in the following format: 
    Predicted Listening Time: [time] seconds
    """


def prompt_template_v2():
    """Adds engagement context (30 seconds threshold) for listening time prediction."""

    return """
    You are a mother enrolled in the ARMMAN Maternal and Child Healthcare Mobile Health program. 
    ARMMAN is a non-governmental organization in India focused on reducing maternal and neonatal mortality among underprivileged communities. 
    Through this program, you receive weekly preventive health information via automated voice messages. 
    While the program has been successful, maintaining consistent engagement is challenging, as many mothers tend to listen less over time. 
    Live service calls from health workers can improve engagement, but their availability is limited.

    Each time step in this simulation represents one week.

    Below is your background and history with the program. Based on this information, decide how long you will listen to the next automated health message.

    - **Your Background:**
      - You enrolled in the program during the {enroll_gest_age} week of your pregnancy.
      - You are {age_category} years old.
      - Your family's monthly income is {income_bracket} in Indian Rupees.
      - Your education level is {education_level}.
      - You speak {language}.
      - You own a {phone_owner} phone.
      - You prefer receiving calls in the {call_slot_preference}.
      - You enrolled in the program through the {channel_type} channel.
      - You are currently in the {enroll_delivery_status} stage of pregnancy.
      - It has been {days_to_first_call} days since you received your first call.
      - You have been pregnant {g} times, with {p} successful births.
      - You have experienced {s} stillbirth(s) and have {l} living child(ren).

    - **Past Behavior:**
      Below is a record of your previous listening behavior (each representing one week):
      - 'Action' indicates whether you received a service call from a health worker to improve your engagement.
      {past_behavior}

    **Key Consideration:** If you listen to the message for more than 30 seconds, it indicates that you are engaged with the program. Listening for less than 30 seconds suggests a lack of engagement.

    **Question:** As you receive the next automated health message, how long will you listen to it?

    Please respond with the predicted listening time in the following format: 
    Predicted Listening Time: [time] seconds
    """


def prompt_template_v3():
    """Omits sociodemographic features, focusing only on past behavior for time prediction."""

    return """
    You are a mother enrolled in the ARMMAN Maternal and Child Healthcare Mobile Health program. Below is your history with the program. Based on this information, decide how long you will listen to the next automated health message.

    Each time step in this simulation represents one week.

    - **Past Behavior:**
      The following is a record of your previous listening behavior (each representing one week):
      - 'Action' indicates whether you received a service call from a health worker aimed at improving your engagement.
      {past_behavior}

    **Question:** As you receive the next automated health message, how long will you listen to it?

    Please respond with the predicted listening time in the following format: 
    Predicted Listening Time: [time] seconds
    """


def prompt_template_v4():
    """Includes full background and notes that actions affect future behavior, not current week."""

    return """
    You are a mother enrolled in the ARMMAN Maternal and Child Healthcare Mobile Health program. Below is your background and history with the program. Based on this information, decide how long you will listen to the next automated health message.

    Each time step in this simulation represents one week.
    - **Your Background:**
      - You enrolled in the program during the {enroll_gest_age} week of your pregnancy.
      - You are {age_category} years old.
      - Your family's monthly income is {income_bracket} in Indian Rupee.
      - Your education level is {education_level}.
      - You speak {language}.
      - You own a {phone_owner} phone.
      - You prefer receiving calls in the {call_slot_preference}.
      - You enrolled in the program through the {channel_type} channel.
      - You are currently in the {enroll_delivery_status} stage of pregnancy.
      - It has been {days_to_first_call} days since you received your first call.
      - You have been pregnant {g} times, with {p} successful births.
      - You have experienced {s} stillbirth(s) and have {l} living child(ren).

    - **Past Behavior:**
      The following is a record of your previous listening behavior (each representing one week):
      {past_behavior}

    **Question:** As you receive the next automated health message, how long will you listen to it?

    Please respond with the predicted listening time in the following format: 
    Predicted Listening Time: [time] seconds
    """


def prompt_template_v5():
    """A minimal prompt focusing only on past behavior for listening time prediction."""

    return """
    Based on your past listening behavior, decide how long you will listen to the next automated health message.

    {past_behavior}

    Please respond with the predicted listening time in the following format:
    Predicted Listening Time: [time] seconds

    **Example Answer:** Predicted Listening Time: 45 seconds
    """


#### Engagement (Binary) Prediction

def bin_prompt_v1():
    """Simplifies the decision to a binary engagement response (##Yes## or ##No##)."""

    return """
    You are a mother enrolled in the ARMMAN Maternal and Child Healthcare Mobile Health program. 
    ARMMAN is a non-governmental organization in India dedicated to reducing maternal and neonatal mortality among underprivileged communities. 
    Through this program, you receive weekly preventive health information via automated voice messages. 
    While the program has been successful, maintaining consistent engagement is challenging, as listenership of any individual mother fluctuates weekly and many mothers tend to listen less over time. 

    In this simulation, each time step represents one week.

    Below is your background and history with the program.

    - **Your Background:**
      - You enrolled in the program during the {enroll_gest_age} week of your pregnancy.
      - You are {age_category} years old.
      - Your family's monthly income is {income_bracket} Indian Rupees.
      - Your education level is {education_level}.
      - You speak {language}.
      - You own a {phone_owner} phone.
      - You prefer receiving calls in the {call_slot_preference} time slot.
      - You enrolled in the program through the {channel_type} channel.
      - You are currently in the {enroll_delivery_status} stage of pregnancy.
      - It has been {days_to_first_call} days since you received your first call.
      - You have been pregnant {g} times, with {p} successful births.
      - You have experienced {s} stillbirth(s) and have {l} living child(ren).

    - **Past Behavior:**
      The following is a record of your previous listening behavior (each representing one week):
      {past_behavior}

    Based on this information, as well as the context of the program and on typical behavior of mothers in India,
    decide whether you will be engaged with the next automated health message.

    **Key Consideration:** Engagement at one week does not imply engagement at the next (and the same for lack of engagement). 
    Engagement should also depend on your specific circumstances that week (e.g. phone availability, schedule, etc.), which may fluctuate.
    Being unable to answer a call that week implies a lack of engagement for that week.

    **Question:** Will you be engaged with the next automated health message?

    Please respond with your final decision in the format: '##Yes##' for engagement or '##No##' for lack of engagement in this week.
    Your response need only contain one of the following: '##Yes##' OR '##No##'. No other text should be included. 
    """


def bin_prompt_v2():
    """Focuses on recent behavior and asks for a binary engagement response."""

    return """
    You are a mother enrolled in the ARMMAN Maternal and Child Healthcare Mobile Health program. 
    This program provides weekly preventive health messages to help reduce maternal and neonatal mortality.
    Below is your background and history with the program. 
    Based on this information, as well as the context of the program and on typical behavior of mothers in India,
    decide whether you will be engaged with the next automated health message.

    Each time step in this simulation represents one week.
    - **Your Background:**
      - You enrolled in the program during the {enroll_gest_age} week of your pregnancy.
      - You are {age_category} years old.
      - Your family's monthly income is {income_bracket} in Indian Rupee.
      - Your education level is {education_level}.
      - You speak {language}.
      - You own a {phone_owner} phone.
      - You prefer receiving calls in the {call_slot_preference}.
      - You enrolled in the program through the {channel_type} channel.
      - You are currently in the {enroll_delivery_status} stage of pregnancy.
      - It has been {days_to_first_call} days since you received your first call.
      - You have been pregnant {g} times, with {p} successful births.
      - You have experienced {s} stillbirth(s) and have {l} living child(ren).

    - **Recent Behavior:**
      The following is a record of your most recent listening behavior:
      {past_behavior}

    **Key Consideration:** If you listen to a message for more than 30 seconds, it indicates that you are engaged with the program. Listening for less than 30 seconds suggests a lack of engagement.

    **Key Consideration:** Engagement at one week does not imply engagement at the next (and the same for lack of engagement). 
    Engagement should also depend on your specific circumstances that week (e.g. phone availability, schedule, etc.), which may fluctuate.
    Being unable to answer a call that week implies a lack of engagement for that week.

    **Question:** Will you be engaged with the next automated health message?

    Please provide the answer in the format: '##Yes##' for engagement OR '##No##' for lack of engagement.
    """


def bin_prompt_v3():
    """A minimal prompt focusing only on past behavior for a binary engagement decision."""

    return """
    Based on the following record of your previous listening behavior and on typical behavior of mothers in India,
    decide whether you will be engaged with the next automated health message.

    - **Past Behavior:**
      The following is a record of your previous listening behavior:
      {past_behavior}

    **Key Consideration:** Engagement at one week does not imply engagement at the next (and the same for lack of engagement). 
    Engagement should also depend on your specific circumstances that week (e.g. phone availability, schedule, etc.), which may fluctuate.
    Being unable to answer a call that week implies a lack of engagement for that week.

    **Question:** Will you be engaged with the next automated health message?

    Please respond with your final decision in the format: '##Yes##' for engagement OR '##No##' for lack of engagement.
    """


def bin_prompt_v4():
    """Emphasizes the role of weekly engagement based on specific circumstances."""

    return """
    You are participating in the ARMMAN Maternal and Child Healthcare Mobile Health program, receiving weekly health updates to support your well-being and that of your child. 
    The program is vital to underprivileged communities in India, where consistent engagement with preventive health information can improve outcomes. 
    However, staying consistently engaged each week can be challenging as various circumstances affect your ability to listen.

    Each week in this simulation represents one opportunity to engage with health messages. Below is your background and recent weekly behavior.

    - **Your Profile:**
      - Enrollment week of pregnancy: {enroll_gest_age}.
      - Age: {age_category}.
      - Family income: {income_bracket} INR.
      - Education: {education_level}.
      - Language: {language}.
      - Phone ownership: {phone_owner}.
      - Preferred call time: {call_slot_preference}.
      - Enrolled through: {channel_type}.
      - Pregnancy stage: {enroll_delivery_status}.
      - Time since first call: {days_to_first_call} days.
      - Pregnancy history: {g} pregnancies, {p} successful births, {s} stillbirth(s), {l} living child(ren).

    - **Recent Weekly Listening Record:**
      {past_behavior}

    **Decision**: Will you engage with the upcoming message, given your unique situation?

    **Please answer** with '##Yes##' for engagement or '##No##' if you will not engage. Provide only one of these responses.
    """


def bin_prompt_v5():
    """Encourages a weekly reflection based on background and recent program experience."""

    return """
    As a mother enrolled in ARMMAN’s Maternal and Child Healthcare Program, you receive weekly health messages to support you during pregnancy and early motherhood. 
    Engaging with these messages helps you stay informed on health practices. 
    Still, it’s common for mothers to listen less over time due to daily demands and circumstances.

    Each week in this simulation is a chance to decide if you will engage with the health message. 
    Below is your background and recent behavior with the program.

    - **Your Background Information:**
      - Enrolled in week {enroll_gest_age} of pregnancy.
      - Age: {age_category}.
      - Monthly income: {income_bracket} INR.
      - Education: {education_level}.
      - Language: {language}.
      - Owns a {phone_owner} phone.
      - Preferred time for calls: {call_slot_preference}.
      - Enrollment channel: {channel_type}.
      - Current pregnancy stage: {enroll_delivery_status}.
      - First call received {days_to_first_call} days ago.
      - Pregnancy history: {g} pregnancies, {p} live births, {s} stillbirth(s), {l} child(ren).

    - **Behavioral History:**
      Below is a record of your recent listening behavior:
      {past_behavior}

    **Key Note**: Engagement each week depends on factors such as phone availability, schedule, or personal demands, which may vary. 
    If you do not answer the call in a given week, this is marked as non-engagement for that week.

    **Question**: Will you engage with the next health message?

    Respond with '##Yes##' to indicate engagement, or '##No##' if you will not engage.
    """


#### Result of Action (Binary) Predictions (intervention cases)


def action_prompt_v1():
    """Binary engagement response (##Yes## or ##No##) based on live service call."""

    return """
    You are a mother enrolled in the ARMMAN Maternal and Child Healthcare Mobile Health program. 
    ARMMAN is a non-governmental organization in India dedicated to reducing maternal and neonatal mortality among underprivileged communities. 
    Through this program, you receive weekly preventive health information via automated voice messages. 
    While the program has been successful, maintaining consistent engagement is challenging, as listenership of any individual mother fluctuates weekly and many mothers tend to listen less over time. 

    In this simulation, each time step represents one week.

    Below is your background and history with the program.

    - **Your Background:**
      - You enrolled in the program during the {enroll_gest_age} week of your pregnancy.
      - You are {age_category} years old.
      - Your family's monthly income is {income_bracket} Indian Rupees.
      - Your education level is {education_level}.
      - You speak {language}.
      - You own a {phone_owner} phone.
      - You prefer receiving calls in the {call_slot_preference} time slot.
      - You enrolled in the program through the {channel_type} channel.
      - You are currently in the {enroll_delivery_status} stage of pregnancy.
      - It has been {days_to_first_call} days since you received your first call.
      - You have been pregnant {g} times, with {p} successful births.
      - You have experienced {s} stillbirth(s) and have {l} living child(ren).

    - **Past Behavior:**
      The following is a record of your previous listening behavior (each representing one week):
      {past_behavior}

    This week, instead of an automated health message, you receive a live service call from a health worker aimed at improving your engagement with the program.

    Based on this information, as well as the context of the program and on typical behavior of mothers in India,
    decide whether you will be engaged with this live call and its health message.

    **Key Consideration:** Engagement at one week does not imply engagement at the next (and the same for lack of engagement). 
    Engagement should also depend on your specific circumstances that week (e.g. phone availability, schedule, etc.), which may fluctuate.
    Being unable to answer a call that week implies a lack of engagement for that week.

    **Question:** Will you be engaged with this week's health message?

    Please respond with your final decision in the format: '##Yes##' for engagement or '##No##' for lack of engagement in this week.
    Your response need only contain one of the following: '##Yes##' OR '##No##'. No other text should be included. 
    """


def action_prompt_v2():
    """Focuses on recent behavior and asks for a binary engagement response to a live service call."""

    return """
    You are a mother enrolled in the ARMMAN Maternal and Child Healthcare Mobile Health program. 
    This program provides weekly preventive health messages to help reduce maternal and neonatal mortality.
    Below is your background and history with the program. 
    Based on this information, as well as the context of the program and on typical behavior of mothers in India,
    decide whether you will be engaged with the next health message you receive via a live service call.

    Each time step in this simulation represents one week.
    - **Your Background:**
      - You enrolled in the program during the {enroll_gest_age} week of your pregnancy.
      - You are {age_category} years old.
      - Your family's monthly income is {income_bracket} in Indian Rupee.
      - Your education level is {education_level}.
      - You speak {language}.
      - You own a {phone_owner} phone.
      - You prefer receiving calls in the {call_slot_preference}.
      - You enrolled in the program through the {channel_type} channel.
      - You are currently in the {enroll_delivery_status} stage of pregnancy.
      - It has been {days_to_first_call} days since you received your first call.
      - You have been pregnant {g} times, with {p} successful births.
      - You have experienced {s} stillbirth(s) and have {l} living child(ren).

    - **Recent Behavior:**
      The following is a record of your most recent listening behavior:
      {past_behavior}

    **Key Consideration:** If you listen to the call and its message for more than 30 seconds, it indicates that you are engaged with the program. Listening for less than 30 seconds suggests a lack of engagement.

    **Key Consideration:** Engagement at one week does not imply engagement at the next (and the same for lack of engagement). 
    Engagement should also depend on your specific circumstances that week (e.g. phone availability, schedule, etc.), which may fluctuate.
    Being unable to answer a call that week implies a lack of engagement for that week.

    This week, instead of an automated health message, you receive a live service call from a health worker aimed at improving your engagement with the program.

    **Question:** Will you be engaged with this week's health message?

    Please provide the answer in the format: '##Yes##' for engagement OR '##No##' for lack of engagement.
    """


def action_prompt_v3():
    """A minimal prompt focusing only on past behavior (not sociodemographic information) for a binary engagement decision based on a live service call."""

    return """
    The following is a record of your previous listening behavior to automated health messages you have received weekly by phone.

    - **Past Behavior:**
      The following is a record of your previous listening behavior:
      {past_behavior}

    Based on this record, and on typical behavior of mothers in India, decide whether you will be engaged with the next health message you receive by phone.
    Instead of the usual automated message, you will receive a live service call from a health worker this week.

    **Key Consideration:** Engagement at one week does not imply engagement at the next (and the same for lack of engagement). 
    Engagement should also depend on your specific circumstances that week (e.g. phone availability, schedule, etc.), which may fluctuate.
    Being unable to answer a call that week implies a lack of engagement for that week.

    **Question:** Will you be engaged with this live service call?

    Please respond with your final decision in the format: '##Yes##' for engagement OR '##No##' for lack of engagement.
    """


def action_prompt_v4():
    """Emphasizes the role of weekly engagement based on specific circumstances. Asks for a response to a live service call."""

    return """
    You are participating in the ARMMAN Maternal and Child Healthcare Mobile Health program, receiving weekly health messages to support your well-being and that of your child. 
    The program is vital to underprivileged communities in India, where consistent engagement with preventive health information can improve outcomes. 
    However, staying consistently engaged each week can be challenging as various circumstances affect your ability to listen.

    Each week in this simulation represents one opportunity to engage with health messages. Below is your background and recent weekly behavior.

    - **Your Profile:**
      - Enrollment week of pregnancy: {enroll_gest_age}.
      - Age: {age_category}.
      - Family income: {income_bracket} INR.
      - Education: {education_level}.
      - Language: {language}.
      - Phone ownership: {phone_owner}.
      - Preferred call time: {call_slot_preference}.
      - Enrolled through: {channel_type}.
      - Pregnancy stage: {enroll_delivery_status}.
      - Time since first call: {days_to_first_call} days.
      - Pregnancy history: {g} pregnancies, {p} successful births, {s} stillbirth(s), {l} living child(ren).

    - **Recent Weekly Listening Record:**
      {past_behavior}

    This week, you will receive a live call from a health worker providing updates about maternal and child health.

    **Decision**: Will you engage with the upcoming call, given your unique situation?

    **Please answer** with '##Yes##' for engagement or '##No##' if you will not engage. Provide only one of these responses.
    """


def action_prompt_v5():
    """Encourages a weekly reflection based on background and recent program experience. Asks for a response to a live service call."""

    return """
    As a mother enrolled in ARMMAN’s Maternal and Child Healthcare Program, you receive weekly health messages to support you during pregnancy and early motherhood. 
    Engaging with these messages helps you stay informed on health practices. 
    Still, it’s common for mothers to listen less over time due to daily demands and circumstances.

    Each week in this simulation is a new chance to decide if you will engage with that week's health message. 
    Below is your background and recent behavior with the program.

    - **Your Background Information:**
      - Enrolled in week {enroll_gest_age} of pregnancy.
      - Age: {age_category}.
      - Monthly income: {income_bracket} INR.
      - Education: {education_level}.
      - Language: {language}.
      - Owns a {phone_owner} phone.
      - Preferred time for calls: {call_slot_preference}.
      - Enrollment channel: {channel_type}.
      - Current pregnancy stage: {enroll_delivery_status}.
      - First call received {days_to_first_call} days ago.
      - Pregnancy history: {g} pregnancies, {p} live births, {s} stillbirth(s), {l} child(ren).

    - **Behavioral History:**
      Below is a record of your recent listening behavior:
      {past_behavior}

    **Key Note**: Engagement each week depends on factors such as phone availability, schedule, or personal demands, which may vary. 
    If you do not answer the call in a given week, this is marked as non-engagement for that week.
    This week, you will receive a live call from a health worker providing maternal and child health updates.

    **Question**: Will you engage with this week's call?

    Respond with '##Yes##' to indicate engagement, or '##No##' if you will not engage.
    """


### No-data Prompts 


def no_data_prompt():
    """Uses only background features (no past behaviour => no data, new programme) to predict future listening time."""
    
    return """
    You are a mother enrolled in the ARMMAN Maternal and Child Healthcare Mobile Health program. 
    Based on your sociodemographic information, decide how long you will listen to the next automated health message.

    Each time step in this simulation represents one week.
    
    - **Your Background:**
      - You enrolled in the program during the {enroll_gest_age} week of your pregnancy.
      - You are {age_category} years old.
      - Your family's monthly income is {income_bracket} in Indian Rupees.
      - Your education level is {education_level}.
      - You speak {language}.
      - You own a {phone_owner} phone.
      - You prefer receiving calls in the {call_slot_preference}.
      - You enrolled in the program through the {channel_type} channel.
      - You are currently in the {enroll_delivery_status} stage of pregnancy.
      - It has been {days_to_first_call} days since you received your first call.
      - You have been pregnant {g} times, with {p} successful births.
      - You have experienced {s} stillbirth(s) and have {l} living child(ren).

    **Question:** As you receive the next automated health message, how long will you listen to it?

    Please respond with the predicted listening time in the following format: 
    Predicted Listening Time: [time] seconds
    """


## Starting Week Prompts (t = 0)

def starting_prompt_v1():
    """Uses only background features to predict listening time in the starting week in the program."""

    return """
            You are a mother enrolled in the ARMMAN Maternal and Child Healthcare Mobile Health program. 
            ARMMAN is a non-governmental organization in India dedicated to reducing maternal and neonatal mortality among underprivileged communities. 
            Through this program, you receive weekly preventive health information via automated voice messages. 
            While the program has been successful, maintaining consistent engagement is challenging, as listenership of any individual mother fluctuates weekly and many mothers tend to listen less over time. 

            This is your starting week in the simulation as you have just enrolled in the program. 

            - **Your Background:**
            - You enrolled in the program during the {enroll_gest_age} week of your pregnancy.
            - You are {age_category} years old.
            - Your family's monthly income is {income_bracket} Indian Rupees.
            - Your education level is {education_level}.
            - You speak {language}.
            - You own a {phone_owner} phone.
            - You prefer receiving calls in the {call_slot_preference} time slot.
            - You enrolled in the program through the {channel_type} channel.
            - You are currently in the {enroll_delivery_status} stage of pregnancy.
            - It has been {days_to_first_call} days since you received your first call.
            - You have been pregnant {g} times, with {p} successful births.
            - You have experienced {s} stillbirth(s) and have {l} living child(ren).

            **Key Consideration:** 
            Engagement should also depend on your specific circumstances that week (e.g. phone availability, schedule, etc.), which may fluctuate.
            Being unable to answer a call that week implies a lack of engagement for that week.

            **Question:** As you receive the first automated health message, how long will you listen to it?

            Please respond with the predicted listening time in the following format: 
            Predicted Listening Time: [time] seconds
            """


def starting_prompt_v2():
    """Uses only background features to predict binary engagement in the starting week in the program."""

    return """
            You are a mother enrolled in the ARMMAN Maternal and Child Healthcare Mobile Health program. 
            ARMMAN is a non-governmental organization in India dedicated to reducing maternal and neonatal mortality among underprivileged communities. 
            Through this program, you receive weekly preventive health information via automated voice messages. 
            While the program has been successful, maintaining consistent engagement is challenging, as many mothers tend to listen less over time. 

            This is your starting week in the simulation as you have just enrolled in the program. 

            - **Your Background:**
            - You enrolled in the program during the {enroll_gest_age} week of your pregnancy.
            - You are {age_category} years old.
            - Your family's monthly income is {income_bracket} Indian Rupees.
            - Your education level is {education_level}.
            - You speak {language}.
            - You own a {phone_owner} phone.
            - You prefer receiving calls in the {call_slot_preference} time slot.
            - You enrolled in the program through the {channel_type} channel.
            - You are currently in the {enroll_delivery_status} stage of pregnancy.
            - It has been {days_to_first_call} days since you received your first call.
            - You have been pregnant {g} times, with {p} successful births.
            - You have experienced {s} stillbirth(s) and have {l} living child(ren).

            **Key Consideration:** If you listen to a message for more than 30 seconds, it indicates that you are engaged with the program. Listening for less than 30 seconds suggests a lack of engagement.

            **Key Consideration:** 
            Engagement should also depend on your specific circumstances that week (e.g. phone availability, schedule, etc.), which may fluctuate.
            Being unable to answer a call that week implies a lack of engagement for that week.

            **Question:** Will you be engaged with the first automated health message?

            Please provide the answer in the format: '##Yes##' for engagement OR '##No##' for lack of engagement.
            """


def starting_prompt_v3():
    """A simplified prompt for the starting week with no past behavior."""

    return """
            In this simulation, this is your starting week, as you have just enrolled in the program.

            **Question:** Will you be engaged with the first automated health message?

            Please respond your final decision in the format: '##Yes##' for engagement or '##No##' for lack of engagement.
            """


def starting_prompt_v4():
    """Emphasizes the value of ARMMAN's program and asks for an initial engagement response."""

    return """
            Welcome to the ARMMAN Maternal and Child Healthcare Mobile Health program. 
            ARMMAN is a non-profit organization in India focused on supporting maternal and neonatal health in underprivileged communities. 
            Through this program, you will receive weekly health guidance tailored to help you and your family.

            As this is your starting week in the simulation, you are receiving the first automated health message.

            - **Your Profile:**
              - Enrollment in the {enroll_gest_age} week of pregnancy.
              - Age: {age_category} years.
              - Monthly family income: {income_bracket} INR.
              - Education: {education_level}.
              - Primary language: {language}.
              - Phone type: {phone_owner}.
              - Preferred call time: {call_slot_preference}.
              - Enrollment channel: {channel_type}.
              - Pregnancy stage: {enroll_delivery_status}.
              - Days since first call: {days_to_first_call}.
              - Pregnancy history: {g} pregnancies, {p} live births, {s} stillbirth(s), {l} living child(ren).

            **Key Note**: Engaging with this program can be beneficial for you and your baby’s health. 
            However, whether you engage with each message may depend on your schedule, phone availability, and specific needs at any given time.

            **Question:** Will you listen to and engage with this first health message?

            Please provide your answer in the format '##Yes##' for engagement or '##No##' for lack of engagement.
            """


def starting_prompt_v5():
    """A concise prompt that encourages the participant to consider the importance of their first engagement decision."""

    return """
            You are beginning your journey in the ARMMAN Maternal and Child Healthcare Mobile Health program. 
            Each week, you will receive an important health message designed to support you through pregnancy and early motherhood.

            - **Your Background Information:**
              - Enrollment in week {enroll_gest_age} of pregnancy.
              - Age: {age_category}.
              - Monthly family income: {income_bracket} INR.
              - Education level: {education_level}.
              - Language: {language}.
              - Type of phone: {phone_owner}.
              - Preferred time for calls: {call_slot_preference}.
              - Program enrollment through: {channel_type}.
              - Pregnancy stage: {enroll_delivery_status}.
              - Days since first call: {days_to_first_call}.
              - History: {g} pregnancies, {p} live births, {s} stillbirth(s), and {l} child(ren) currently.

            **Reminder**: Engaging each week depends on your unique situation that day—phone availability, schedule, and personal priorities may all play a role. 
            Listening for more than 30 seconds shows engagement, while anything less implies no engagement.

            **Question:** As you receive this first health message, will you choose to / be able to engage with it?

            Respond only with '##Yes##' if you will engage or '##No##' if you will not engage.
            """


def starting_prompt_v6():
    """Highlights the importance of the participant's decision and engagement with the program."""

    return """
            Welcome to the ARMMAN Maternal and Child Healthcare Mobile Health program. 
            As a participant, each weekly health message you receive supports your well-being during pregnancy and early motherhood.

            Your engagement with the program, starting from this first week, can provide vital health information to support you and your family. 

            - **Your Profile**:
              - Enrollment in the {enroll_gest_age} week of pregnancy.
              - Age: {age_category} years.
              - Monthly family income: {income_bracket} INR.
              - Education: {education_level}.
              - Language spoken: {language}.
              - Type of phone: {phone_owner}.
              - Preferred time for calls: {call_slot_preference}.
              - Enrollment through: {channel_type}.
              - Pregnancy stage: {enroll_delivery_status}.
              - Days since your first call: {days_to_first_call}.
              - Pregnancy history: {g} pregnancies, {p} successful births, {s} stillbirth(s), and {l} living child(ren).

            **Key Consideration**: Engagement may depend on factors like your schedule and phone availability, which may vary each week. 
            If you don’t answer a call, this counts as non-engagement for that week.

            **Question**: As you receive this first health message, will you choose to engage with it?

            **Response**: Please reply with '##Yes##' if you’ll engage or '##No##' if you won’t engage this week.
            """


## Starting Week (t = 0) Prompts WITH action (intervention service call)


def starting_action_prompt_v1():
    """Uses only background features to predict binary engagement in the starting week in the program when given a live service call."""

    return """
            You are a mother enrolled in the ARMMAN Maternal and Child Healthcare Mobile Health program. 
            ARMMAN is a non-governmental organization in India dedicated to reducing maternal and neonatal mortality among underprivileged communities. 
            Through this program, you receive weekly preventive health information via automated voice messages. This week, instead of an automated message, you will receive a live service call from a health worker.

            This is your starting week in the simulation as you have just enrolled in the program. 

            - **Your Background:**
            - You enrolled in the program during the {enroll_gest_age} week of your pregnancy.
            - You are {age_category} years old.
            - Your family's monthly income is {income_bracket} Indian Rupees.
            - Your education level is {education_level}.
            - You speak {language}.
            - You own a {phone_owner} phone.
            - You prefer receiving calls in the {call_slot_preference} time slot.
            - You enrolled in the program through the {channel_type} channel.
            - You are currently in the {enroll_delivery_status} stage of pregnancy.
            - It has been {days_to_first_call} days since you received your first call.
            - You have been pregnant {g} times, with {p} successful births.
            - You have experienced {s} stillbirth(s) and have {l} living child(ren).

            **Key Consideration:** If you listen to the call or message for more than 30 seconds, it indicates that you are engaged with the program. Listening for less than 30 seconds suggests a lack of engagement.

            **Key Consideration:** 
            Engagement should also depend on your specific circumstances that week (e.g. phone availability, schedule, etc.), which may fluctuate.
            Being unable to answer a call that week implies a lack of engagement for that week.

            You will receive a live service call from a health worker this week, in your first week in the program.
            **Question:** Will you be engaged with the first live service call?

            Please provide the answer in the format: '##Yes##' for engagement OR '##No##' for lack of engagement.
            """


def starting_action_prompt_v2():
    """A simplified prompt for a live service call in the starting week with no past behavior."""

    return """
            In this simulation, this is your starting week, as you have just enrolled in the program. 
            Instead of the usual automated message, you will receive a live service call from a health worker this week.

            **Question:** Will you be engaged with the first live service call?

            Please respond with your final decision in the format: '##Yes##' for engagement or '##No##' for lack of engagement.
            """


def starting_action_prompt_v3():
    """Emphasizes the value of ARMMAN's program and asks for an initial engagement response to a live service call."""

    return """
            Welcome to the ARMMAN Maternal and Child Healthcare Mobile Health program. 
            ARMMAN is a non-profit organization in India focused on supporting maternal and neonatal health in underprivileged communities. 
            Through this program, you will receive weekly health guidance tailored to help you and your family.

            As this is your starting week in the simulation, instead of the usual automated message, you will receive your first health guidance through a live service call from a health worker."

            - **Your Profile:**
              - Enrollment in the {enroll_gest_age} week of pregnancy.
              - Age: {age_category} years.
              - Monthly family income: {income_bracket} INR.
              - Education: {education_level}.
              - Primary language: {language}.
              - Phone type: {phone_owner}.
              - Preferred call time: {call_slot_preference}.
              - Enrollment channel: {channel_type}.
              - Pregnancy stage: {enroll_delivery_status}.
              - Days since first call: {days_to_first_call}.
              - Pregnancy history: {g} pregnancies, {p} live births, {s} stillbirth(s), {l} living child(ren).

            **Key Note**: Engaging with this program can be beneficial for your and your baby’s health. 
            However, whether you engage with each message may depend on your schedule, phone availability, and specific needs at any given time.

            **Question:** Will you listen to and engage with this first health message?

            Please provide your answer in the format '##Yes##' for engagement or '##No##' for lack of engagement.
            """


def starting_action_prompt_v4():
    """A concise prompt that encourages the participant to consider the importance of their first engagement decision given a live service call."""

    return """
            You are beginning your journey in the ARMMAN Maternal and Child Healthcare Mobile Health program. 
            Each week, you will receive an important health message designed to support you through pregnancy and early motherhood.
            For your first week, instead of an automated message, you will receive this guidance through a live service call from a health worker.

            - **Your Background Information:**
              - Enrollment in week {enroll_gest_age} of pregnancy.
              - Age: {age_category}.
              - Monthly family income: {income_bracket} INR.
              - Education level: {education_level}.
              - Language: {language}.
              - Type of phone: {phone_owner}.
              - Preferred time for calls: {call_slot_preference}.
              - Program enrollment through: {channel_type}.
              - Pregnancy stage: {enroll_delivery_status}.
              - Days since first call: {days_to_first_call}.
              - History: {g} pregnancies, {p} live births, {s} stillbirth(s), and {l} child(ren) currently.

            **Reminder**: Engaging each week depends on your unique situation that day—phone availability, schedule, and personal priorities may all play a role. 
            Listening for more than 30 seconds shows engagement, while anything less (including not being able to answer the phone) implies no engagement.

            **Question:** As you receive this first health message, will you choose to / be able to engage with it?

            Respond only with '##Yes##' if you will engage or '##No##' if you will not engage.
            """


def starting_action_prompt_v5():
    """Highlights the importance of the participant's decision and engagement with the program. Given a live service call in week 1."""

    return """
            Welcome to the ARMMAN Maternal and Child Healthcare Mobile Health program. 
            As a participant, each weekly health message you receive supports your well-being during pregnancy and early motherhood.

            Your engagement with the program, starting from this first week, can provide vital health information to support you and your family. 

            - **Your Profile**:
              - Enrollment in the {enroll_gest_age} week of pregnancy.
              - Age: {age_category} years.
              - Monthly family income: {income_bracket} INR.
              - Education: {education_level}.
              - Language spoken: {language}.
              - Type of phone: {phone_owner}.
              - Preferred time for calls: {call_slot_preference}.
              - Enrollment through: {channel_type}.
              - Pregnancy stage: {enroll_delivery_status}.
              - Days since your first call: {days_to_first_call}.
              - Pregnancy history: {g} pregnancies, {p} successful births, {s} stillbirth(s), and {l} living child(ren).

            **Key Consideration**: Engagement may depend on factors like your schedule and phone availability, which may vary each week. 
            If you don’t answer a call, this counts as non-engagement for that week.

            **Key Consideration**: "For your first week, instead of an automated message, you will receive a live service call from a health worker.
            In subsequent weeks you will receive automated health messages.

            **Question**: As you receive this first health message, will you engage with it?

            **Response**: Please reply with '##Yes##' if you’ll engage or '##No##' if you won’t engage this week.
            """
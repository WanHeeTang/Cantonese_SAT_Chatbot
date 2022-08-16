// MessageParser starter code
class MessageParser {
  constructor(actionProvider, state) {
    this.actionProvider = actionProvider;
    this.state = state;
  }

  // This method is called inside the chatbot when it receives a message from the user.
  parse(message) {
    // Add message opject, speech bubble

    // Case: User has not provided id yet
    if (this.state.username == null) {
      return this.actionProvider.askForPassword(message);
    } else if (this.state.password == null) {
      return this.actionProvider.updateUserID(this.state.username, message);
    } else if (this.state.askingForProtocol && parseInt(message) >= 1 && parseInt(message) <= 20) {

      const choice_info = {
        user_id: this.state.userState,
        session_id: this.state.sessionID,
        user_choice: message,
        input_type: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
      };
      this.actionProvider.stopAskingForProtocol()

      return this.actionProvider.sendRequest(choice_info);
    } else if (this.state.askingForProtocol && (parseInt(message) < 1 || parseInt(message) > 20)) {
      return this.actionProvider.askForProtocol()
    }
    
    else if (message.toLowerCase() === "信曦"  || message.toLowerCase() === "權叔" 
    || message.toLowerCase() === "霞姨" || message.toLowerCase() === "偉文" || message.toLowerCase() === "子娟" || message.toLowerCase() === "是" || message.toLowerCase() === "否" || 
    message.toLowerCase() === "繼續" || message.toLowerCase() === "好啊" || message.toLowerCase() === "不用了" || message.toLowerCase() === "對，有特定事情導致" || message.toLowerCase() === "沒有，只是一種感覺" || message.toLowerCase() === "最近發生" || 
    message.toLowerCase() === "好耐之前" || message.toLowerCase() === "有" || message.toLowerCase() === "沒有" || message.toLowerCase() === "好" || message.toLowerCase() === "唔好"
    || message.toLowerCase() === "會" || message.toLowerCase() === "唔會" || message.toLowerCase() === "我覺得好咗" || message.toLowerCase() === "我覺得差咗" || message.toLowerCase() === "我覺得冇變"
    || message.toLowerCase() === "好啊（繼續其他練習）" || message.toLowerCase() === "好啊（重新開始）" || message.toLowerCase() === "不用了（結束）") {
      let input_type = "Protocol";
      // console.log(input_type)
      const currentOptionToShow = this.state.currentOptionToShow
      // console.log(currentOptionToShow)
      // Case: user types when they enter text instead of selecting an option
      if ((currentOptionToShow === "Continue" && message !== "Continue") ||
      (currentOptionToShow === "Emotion" && (message !== "Happy" && message !== "Sad" && message !== "Angry" && message !== "Neutral")) ||
      (currentOptionToShow === "RecentDistant" && (message !== "Recent" && message !== "Distant")) ||
      (currentOptionToShow === "Feedback" && (message !== "Better" && message !== "Worse" && message !== "No change")) ||
      (currentOptionToShow === "Protocol" && (!this.state.protocols.includes(message))) ||
      (currentOptionToShow === "YesNo" && (message !== "Yes" && message !== "No"))
      ) {
        // copy last message when the user does not select an option button.
        this.actionProvider.copyLastMessage()
      } else {
        const choice_info = {
          user_id: this.state.userState,
          session_id: this.state.sessionID,
          user_choice: message,
          input_type: input_type,
        };
        return this.actionProvider.sendRequest(choice_info);
    }}


    else {
      let input_type = null;
      if (this.state.inputType.length === 1) {
        input_type = this.state.inputType[0]
      } else {
        input_type = this.state.inputType
      }
      const currentOptionToShow = this.state.currentOptionToShow
      // Case: user types when they enter text instead of selecting an option
      if ((currentOptionToShow === "Continue" && message !== "Continue") ||
        (currentOptionToShow === "Emotion" && (message !== "Happy" && message !== "Sad" && message !== "Angry" && message !== "Neutral")) ||
        (currentOptionToShow === "RecentDistant" && (message !== "Recent" && message !== "Distant")) ||
        (currentOptionToShow === "Feedback" && (message !== "Better" && message !== "Worse" && message !== "No change")) ||
        (currentOptionToShow === "Protocol" && (!this.state.protocols.includes(message))) ||
        (currentOptionToShow === "YesNo" && (message !== "Yes" && message !== "No"))
      ) {
        this.actionProvider.copyLastMessage()
      } else {
        const choice_info = {
          user_id: this.state.userState,
          session_id: this.state.sessionID,
          user_choice: message,
          input_type: input_type,
        };
        return this.actionProvider.sendRequest(choice_info);
      }
    }

  }
}

export default MessageParser;

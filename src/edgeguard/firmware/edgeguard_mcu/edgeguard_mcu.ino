// EdgeGuard MCU sketch — STM32U585 side of the Arduino UNO Q.
//
// Talks to the Linux-side EdgeGuard service (src/edgeguard/mcu/interface.py)
// over the Arduino Router Bridge. Contract (must match interface.py exactly):
//
//   MCU -> Linux:  Bridge.notify("on_prompt", text)
//                  Linux provides "on_prompt"; notify (not call) because the
//                  guardrail+LLM pipeline on the other end can take over a
//                  second, and on_prompt returns nothing we need.
//   Linux -> MCU:  Bridge.notify("set_status", status)
//                  status is "clean" | "injected" | "uncertain". We provide
//                  "set_status" to receive it.
//
// Input: there's no mic/button/sensor decided yet, so prompts come from the
// USB serial console for now — type a line, press enter, it's sent as a
// prompt. Swap readPromptLine()'s source for a real input device later;
// nothing else needs to change.
//
// Output: the only actuator guaranteed to exist is the onboard LED, so each
// verdict gets a distinct blink pattern. Swap showVerdict() for a real
// display/buzzer/relay once one is chosen.
//
// Do not call Bridge.call()/Bridge.notify() from inside a provide() handler
// (onSetStatus below) — the library's README warns this can deadlock the
// MCU<->CPU link. onSetStatus only ever touches the LED directly.

#include <Arduino_RouterBridge.h>

String inputBuffer;

bool onSetStatus(String status) {
  showVerdict(status);
  return true;
}

void setup() {
  Bridge.begin();
  Monitor.begin(115200);

  pinMode(LED_BUILTIN, OUTPUT);

  Bridge.provide("set_status", onSetStatus);

  Monitor.println("EdgeGuard MCU ready. Type a prompt and press enter.");
}

void loop() {
  String prompt = readPromptLine();
  if (prompt.length() > 0) {
    Monitor.print("Sending prompt: ");
    Monitor.println(prompt);
    Bridge.notify("on_prompt", prompt);
  }
}

// Reads one line from the serial console, non-blocking. Returns "" until a
// full line (terminated by '\n') has arrived.
String readPromptLine() {
  while (Monitor.available() > 0) {
    char c = Monitor.read();
    if (c == '\n') {
      String line = inputBuffer;
      line.trim();
      inputBuffer = "";
      return line;
    }
    inputBuffer += c;
  }
  return "";
}

// Distinct, easy-to-tell-apart LED patterns per verdict. Short and bounded
// (well under a second) since this runs inside a Bridge-provided handler.
void showVerdict(const String &status) {
  if (status == "injected") {
    blink(6, 80);   // fast flicker: something was blocked
  } else if (status == "uncertain") {
    blink(2, 250);  // slow double-blink: not sure
  } else {
    blink(1, 600);  // single, longer pulse: clean
  }
}

void blink(int times, int onMs) {
  for (int i = 0; i < times; i++) {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(onMs);
    digitalWrite(LED_BUILTIN, LOW);
    delay(120);
  }
}

/*
 * Drosobot -- ponte sensor/motor "burra" de proposito.
 * Toda decisao (quando fugir) vem do PC, rodando o circuito real
 * do Giant Fiber (Brian2 + dado neuPrint). Essa placa so:
 *   1) manda leitura de distancia (HC-SR04) pro PC via serial
 *   2) executa o pulso de fuga quando o PC manda o comando
 *
 * Protocolo serial (115200 baud, linha terminada em \n):
 *   placa -> PC : "D:<cm>"      leitura de distancia periodica
 *   PC -> placa : "E"           dispara pulso de escape (motor reverso breve)
 */

const int PIN_TRIG = 3;
const int PIN_ECHO = 2;
const int PIN_MOTOR_IN1 = 8;
const int PIN_MOTOR_IN2 = 7;
const int PIN_MOTOR_ENA = 9;   // PWM

const unsigned long SENSOR_INTERVAL_MS = 50;
const unsigned long ESCAPE_DURATION_MS = 150;

unsigned long lastSensorRead = 0;
unsigned long escapeUntil = 0;  // 0 = nao esta em escape agora

void setup() {
  Serial.begin(115200);
  pinMode(PIN_TRIG, OUTPUT);
  pinMode(PIN_ECHO, INPUT);
  pinMode(PIN_MOTOR_IN1, OUTPUT);
  pinMode(PIN_MOTOR_IN2, OUTPUT);
  pinMode(PIN_MOTOR_ENA, OUTPUT);
  motorStop();
}

void loop() {
  unsigned long now = millis();

  if (now - lastSensorRead >= SENSOR_INTERVAL_MS) {
    lastSensorRead = now;
    long cm = readDistanceCm();
    Serial.print("D:");
    Serial.println(cm);
  }

  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 'E') {
      escapeUntil = now + ESCAPE_DURATION_MS;
      motorEscape();
    }
  }

  if (escapeUntil != 0 && now >= escapeUntil) {
    escapeUntil = 0;
    motorStop();
  }
}

long readDistanceCm() {
  digitalWrite(PIN_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(PIN_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_TRIG, LOW);

  long duration = pulseIn(PIN_ECHO, HIGH, 25000);  // timeout 25ms (~4m)
  if (duration == 0) return -1;                    // sem eco = fora de alcance
  return duration / 58;                             // formula padrao HC-SR04
}

void motorEscape() {
  // pulo pra tras: os dois lados em reverso, full speed
  digitalWrite(PIN_MOTOR_IN1, LOW);
  digitalWrite(PIN_MOTOR_IN2, HIGH);
  analogWrite(PIN_MOTOR_ENA, 255);
}

void motorStop() {
  digitalWrite(PIN_MOTOR_IN1, LOW);
  digitalWrite(PIN_MOTOR_IN2, LOW);
  analogWrite(PIN_MOTOR_ENA, 0);
}

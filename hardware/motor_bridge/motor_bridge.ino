/*
 * Drosobot -- ponte sensor/motor "burra" de proposito.
 * Toda decisao (fugir ou virar) vem do PC, rodando os circuitos reais
 * (Giant Fiber + Optomotor, Brian2 + dado neuPrint). Essa placa so:
 *   1) manda leitura de distancia (HC-SR04) pro PC via serial
 *   2) executa o comando de motor que o PC mandar
 *
 * Drive diferencial, 2 motores (L298N):
 *   motor esquerdo: IN1/IN2/ENA
 *   motor direito : IN3/IN4/ENB
 *
 * Protocolo serial (115200 baud):
 *   placa -> PC : "D:<cm>"   leitura de distancia periodica (a cada 50ms)
 *   PC -> placa : "E"        escape (os dois motores em reverso, pulo pra tras)
 *   PC -> placa : "L"        vira esquerda (motor direito gira, esquerdo para -- pivot)
 *   PC -> placa : "R"        vira direita  (motor esquerdo gira, direito para -- pivot)
 *
 * Prioridade: escape sempre interrompe um giro em andamento (igual biologia --
 * reflexo de fuga e privilegiado sobre comportamento em curso).
 */

const int PIN_TRIG = 3;
const int PIN_ECHO = 2;

const int PIN_MOTOR_L_IN1 = 8;
const int PIN_MOTOR_L_IN2 = 7;
const int PIN_MOTOR_L_EN  = 9;   // PWM
const int PIN_MOTOR_R_IN1 = 6;
const int PIN_MOTOR_R_IN2 = 5;
const int PIN_MOTOR_R_EN  = 10;  // PWM

const unsigned long SENSOR_INTERVAL_MS = 50;
const unsigned long ESCAPE_DURATION_MS = 150;
const unsigned long TURN_DURATION_MS = 80;

unsigned long lastSensorRead = 0;
unsigned long escapeUntil = 0;  // 0 = nao esta em escape agora
unsigned long turnUntil = 0;    // 0 = nao esta virando agora
char turnDir = 0;               // 'L' ou 'R'

void setup() {
  Serial.begin(115200);
  pinMode(PIN_TRIG, OUTPUT);
  pinMode(PIN_ECHO, INPUT);
  pinMode(PIN_MOTOR_L_IN1, OUTPUT);
  pinMode(PIN_MOTOR_L_IN2, OUTPUT);
  pinMode(PIN_MOTOR_L_EN, OUTPUT);
  pinMode(PIN_MOTOR_R_IN1, OUTPUT);
  pinMode(PIN_MOTOR_R_IN2, OUTPUT);
  pinMode(PIN_MOTOR_R_EN, OUTPUT);
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
    } else if (cmd == 'L' || cmd == 'R') {
      turnUntil = now + TURN_DURATION_MS;
      turnDir = cmd;
    }
  }

  if (escapeUntil != 0 && now < escapeUntil) {
    motorEscape();                 // escape manda, ignora giro em andamento
  } else {
    escapeUntil = 0;
    if (turnUntil != 0 && now < turnUntil) {
      motorTurn(turnDir);
    } else {
      turnUntil = 0;
      motorStop();
    }
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
  // pulo pra tras: os dois motores em reverso, full speed
  digitalWrite(PIN_MOTOR_L_IN1, LOW);
  digitalWrite(PIN_MOTOR_L_IN2, HIGH);
  analogWrite(PIN_MOTOR_L_EN, 255);
  digitalWrite(PIN_MOTOR_R_IN1, LOW);
  digitalWrite(PIN_MOTOR_R_IN2, HIGH);
  analogWrite(PIN_MOTOR_R_EN, 255);
}

void motorTurn(char dir) {
  // pivot: um lado gira pra frente, o outro para (giro no proprio eixo)
  if (dir == 'L') {
    digitalWrite(PIN_MOTOR_L_IN1, LOW);
    digitalWrite(PIN_MOTOR_L_IN2, LOW);
    analogWrite(PIN_MOTOR_L_EN, 0);
    digitalWrite(PIN_MOTOR_R_IN1, HIGH);
    digitalWrite(PIN_MOTOR_R_IN2, LOW);
    analogWrite(PIN_MOTOR_R_EN, 200);
  } else {
    digitalWrite(PIN_MOTOR_R_IN1, LOW);
    digitalWrite(PIN_MOTOR_R_IN2, LOW);
    analogWrite(PIN_MOTOR_R_EN, 0);
    digitalWrite(PIN_MOTOR_L_IN1, HIGH);
    digitalWrite(PIN_MOTOR_L_IN2, LOW);
    analogWrite(PIN_MOTOR_L_EN, 200);
  }
}

void motorStop() {
  digitalWrite(PIN_MOTOR_L_IN1, LOW);
  digitalWrite(PIN_MOTOR_L_IN2, LOW);
  analogWrite(PIN_MOTOR_L_EN, 0);
  digitalWrite(PIN_MOTOR_R_IN1, LOW);
  digitalWrite(PIN_MOTOR_R_IN2, LOW);
  analogWrite(PIN_MOTOR_R_EN, 0);
}

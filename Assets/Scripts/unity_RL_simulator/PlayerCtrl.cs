using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class State_unityRL {
    public byte[] sensor_data = new byte[181];    // input array : 센서 0~180번

    public float engine = 0;     // output0 : 차량의 속도
    public float turn = 0;       // output1 : 차량의 회전

    //public float reward = 1;     // 보상

    public byte isDone = 0;       // 게임이 끝났는가?
    //38, -65
    public int Goal_X = 35;
    public int Goal_Y = -39;
    public float Goal_heading = Mathf.PI;   //[deg]
    public float posRange = 1.5f;           //[m]
    public float heaingRange = 19.0f;       //[deg] diff range 180 -> 128

    public float vecLen = 0.0f;         //[m]
    public float headingDiff = 0.0f;    //[deg]

    public int episode = 0;      // 에피소드

    public void reset() {
        for (int i = 0; i < sensor_data.Length; i++) {
            sensor_data[i] = 0;
        }

        //reward = 1;
        episode = 0;
    }
}

public class PlayerCtrl : MonoBehaviour {
    private int sensorLength = 181;
    private float[] sensorOutput;

    /// ed: 코드 추가
    public GameObject PrototypeCar;
    public GameMgr gameMgr;

    private SendPacket sendPacket;
    private State_unityRL state_unityRL;
    private Sensor[] sensors;
    private float timeSinceLastCheckpoint;


    /// <summary>
    /// Whether this car is controllable by user input (keyboard).
    /// </summary>
    public bool UseUserInput = false;

    /// <summary>
    /// The movement component of this car.
    /// </summary>
    public PlayerMovement Movement {
        get;
        private set;
    }

    /// <summary>
    /// The current inputs for controlling the CarMovement component.
    /// </summary>
    public double[] CurrentControlInputs {
        get { return Movement.CurrentInputs; }
    }

    /// <summary>
    /// The cached SpriteRenderer of this car.
    /// </summary>
    public SpriteRenderer SpriteRenderer {
        get;
        private set;
    }

    /// ed: 목표지점에 도착했는지 체크하는 변수
    private bool isInGoal = false;
    private bool isFail = false;


    /// ed: 플레이어가 죽은 경우 실행되는 코루틴함수
    IEnumerator StartResetEpisode() {
        yield return new WaitForFixedUpdate();

        this.transform.position = PrototypeCar.transform.position;
        this.transform.rotation = PrototypeCar.transform.rotation;
        Movement.enabled = true;
        timeSinceLastCheckpoint = 0;

        foreach (Sensor s in sensors)
            s.Show();

        this.enabled = true;
        this.isInGoal = true;
        this.isFail = false;

        ResetEpisode();
    }


    /// ed: 데이터를 받으면 실행되는 코루틴함수
    public IEnumerator Action_unityRL(float recv_data) {

        /// ed: socket으로부터 받은 output 데이터는 One Hot Vector이므로
        ///      아래와 같이 분기별로 다른 action sets 들을 만들어줘야합니다
        ///      각 분기별 값은 임의로 지정할 수 있습니다.
        const int omega_idx = 3;
        float[] omega_dot = new float[omega_idx];
        omega_dot[0] = -1.0f;
        omega_dot[1] = 0.0f;
        omega_dot[2] = +1.0f;

        const int acc_idx = 3;
        float[] acc = new float[acc_idx];
        acc[0] = 0.3f;
        acc[1] = 1.5f;
        acc[2] = 3.0f;

        //recv_data 
        int i = (int)((int)recv_data / acc_idx);    // omega_idx
        int j = (int)recv_data - acc_idx * i;       // acc_idx

        Movement.SetInputs(omega_dot[i], acc[j]);   //실제 차 움직이게 한다.
        
        /// ed: 에피소드를 하나 증가시키고
        state_unityRL.episode++;
        yield return new WaitForFixedUpdate();

        /// ed: Input으로 사용할 센서값을 받아오고
        UpdateState();
        /// ed: 차량과 부딪혔는지 검사합니다 
        CheckFail();
    }
    
    
    /// ed: 차량이 부딪혔는지 체크하고 실행되는 함수
    public void CheckFail() {
        /// ed: 차량이 부딪혔다면
        if (isFail) {
            state_unityRL.isDone = 1;
            SocketServer.instance.SendMessage(ConvertData());
            //한 에피소드씩 재생할 때 사용
            StartCoroutine(StartResetEpisode());
            //ResetEpisode();

            
        }
        /// ed: 차량이 안 부딪힌 경우 실행되는 코드
        else
            SocketServer.instance.SendMessage(ConvertData());
    }


    /// ed: socket으로 전송할 센서값을 저장하는 함수
    void UpdateState() {
        int tmp_tr = 4; // 센서는 m단위의 값을 보낸다. 통신을 하는 경우 소숫점이 날라가서 이를 큰 수로 변환을 하여 보낸다.
                        // 최대 25m를 보기 때문에, 4를 곱해 100이하의 수를 가지도록 한다.
        /// ed: socket으로 전송하기 위해 센서들의 값을 저장합니다
        for (int i =0; i < sensors.Length; i++) {
            state_unityRL.sensor_data[i] = (byte)(sensorOutput[i] * tmp_tr);
        }
    }


    private void Awake() {
        Movement = GetComponent<PlayerMovement>();
        sensors = GetComponentsInChildren<Sensor>();
        SpriteRenderer = GetComponent<SpriteRenderer>();

        sendPacket = new SendPacket();
        sendPacket.sensorData = new byte[sensorLength];
        state_unityRL = new State_unityRL();

        sensorOutput = new float[sensors.Length];
    }

    private void Start() {
        /// event 개념을 활용해서 HitCar 이벤트에 Die 함수를 연결한다
        Movement.HitCar += Die;
    }

    private void Update() {
        timeSinceLastCheckpoint += Time.deltaTime;
    }


    void FixedUpdate() {
        /// ed: 센서값을 받아오는 코드
        for (int i = 0; i < sensors.Length; i++)
            sensorOutput[i] = sensors[i].Output;
        
        /// ed: 목표지점을 설정
        Bounds bounds = new Bounds(new Vector3(state_unityRL.Goal_X, state_unityRL.Goal_Y, 0), 
            new Vector3(state_unityRL.posRange, state_unityRL.posRange, 0));

        /// ed: 현재차량이 목표지점(Goal)에 도착하면 아래 코드가 시작된다
        /// 차와 goal간 거리를 나타낸다,
        /// 거리가 너무 크기 때문에 0.07이라는 값을 곱해 크기를 줄인다.
        state_unityRL.vecLen = bounds.SqrDistance(this.transform.position) * 0.07f;
        //if (state_unityRL.vecLen > 127) state_unityRL.vecLen = 127;
        //Debug.Log("vector len : " + state_unityRL.vecLen);

        // 차와 goal간 방향 차이를 나타낸다. [deg]
        float car_heading = Mathf.Abs(Quaternion.ToEulerAngles(this.transform.rotation).z);
        //Debug.Log("vehicle_heading" + car_heading);
        state_unityRL.headingDiff = 180 * (state_unityRL.Goal_heading - car_heading) / Mathf.PI;
        //Debug.Log("heading_diff : " + state_unityRL.headingDiff);

        if (bounds.Contains(this.transform.position)    // 차가 goal위치에 posRange범위에 들어오면, 
            && state_unityRL.headingDiff <= state_unityRL.heaingRange  // 서로의 방향 차이가 heaingRange에 들어오면
            )   // Goal에 도달 했다고 간주
        {
            Debug.Log("Goal!");
            state_unityRL.isDone = 3;
            isInGoal = true;
        }
    }

    /// ed: agent가 벽에 부딪히는 경우 실행되는 함수 (이벤트에 의해 실행된다)
    // Makes this car die (making it unmovable and stops the Agent from calculating the controls for the car).
    private void Die() {
        this.enabled = false;
        Movement.Stop();
        Movement.enabled = false;

        foreach (Sensor s in sensors)
            s.Hide();

        
        this.isFail = true;
    }

    /// ed: 에피소드를 재시작하는 함수
    void ResetEpisode() {
        state_unityRL.reset();
        state_unityRL.isDone = 0;

        SocketServer.instance.SendMessage(ConvertData());
    }


    /// ed: 시뮬레이션 데이터를 socket 전송데이터로 변환하는 함수
    public SendPacket ConvertData() {
        for (int i = 0; i < sensorLength; i++)
            sendPacket.sensorData[i] = state_unityRL.sensor_data[i];

        sendPacket.vecLen = (byte)state_unityRL.vecLen;
        sendPacket.headingDiff = (byte)(state_unityRL.headingDiff);
        sendPacket.isDone = state_unityRL.isDone;

        return sendPacket;
    }
    
}

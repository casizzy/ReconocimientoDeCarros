import cv2
import numpy as np
import os
import mpi  

# ------------- CONFIGURACIÓN -------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "data", "pluma.mp4")

SCALE = 0.5          # bajar tamaño del video (0.5 = 50%)
CALIB_FRAMES = 40    # frames iniciales para calibrar "pluma cerrada"

DELTA_PLUMA = 20     # qué tanto debe cambiar la intensidad para decir "se abrió"
MIN_FRAMES = 5       # mínimo de frames seguidos en estado "abierta" para contar

# ROIs de las dos plumas 
PLUMA1_Y1_F = 0.50   # pluma de adelante 
PLUMA1_Y2_F = 0.58

PLUMA2_Y1_F = 0.40   # pluma de atrás
PLUMA2_Y2_F = 0.48


# --------- FUNCIONES AUXILIARES ----------

def get_gray_small(frame):
    frame_small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
    rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    gray = mpi.RGB2GRAY(rgb).astype(np.uint8)
    return frame_small, gray


def calibrar_baseline(cap, rois, num_frames):
    baseline = np.zeros(len(rois), dtype=np.float32)
    n = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while n < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        _, gray = get_gray_small(frame)

        for i, (y1, y2) in enumerate(rois):
            roi = gray[y1:y2, :]
            baseline[i] += roi.mean()

        n += 1

    if n == 0:
        raise RuntimeError("No se pudieron leer frames para calibrar")

    baseline /= n
    return baseline


# ------------- MAIN ----------------------

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("No se pudo abrir el video:", VIDEO_PATH)
        return

    # leer un frame para conocer tamaño reducido
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el primer frame")
        return

    frame_small, gray = get_gray_small(frame)
    alto, ancho = gray.shape

    # definir ROIs de las dos plumas en pixeles
    pluma1_y1 = int(alto * PLUMA1_Y1_F)
    pluma1_y2 = int(alto * PLUMA1_Y2_F)
    pluma2_y1 = int(alto * PLUMA2_Y1_F)
    pluma2_y2 = int(alto * PLUMA2_Y2_F)

    rois = [(pluma1_y1, pluma1_y2), (pluma2_y1, pluma2_y2)]

    # info del video para limitar a 1 minuto
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps <= 0:
        fps = 30
    max_frames = int(min(total_frames, fps * 60))

    print(f"FPS: {fps:.2f}, frames totales: {total_frames}, usando: {max_frames}")

    # calibrar baseline con plumas cerradas
    print("Calibrando intensidad de plumas cerradas...")
    baseline = calibrar_baseline(cap, rois, CALIB_FRAMES)
    print("Baseline pluma1/pluma2:", baseline)

    # estados
    abierta = [False, False] # estado actual de cada ROI
    frames_abierta = [0, 0] # cuántos frames seguidos lleva abierta
    conteo = [0, 0] # conteo por ROI 

    # kernel de enfoque (sharpen) para usar con mpi.convolucion2D
    sharpen_kernel = np.array([
        [0, -1,  0],
        [-1, 5, -1],
        [0, -1,  0]
    ], dtype=np.float32)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        frame_small, gray = get_gray_small(frame)

        # flags para saber en qué ROI se detectó auto en ESTE frame
        evento_roi = [False, False]

        for idx, (y1, y2) in enumerate(rois):
            roi = gray[y1:y2, :]
            mean_val = roi.mean()

            # condición de "abierta": intensidad se aleja de la baseline
            if not abierta[idx]:
                if abs(mean_val - baseline[idx]) > DELTA_PLUMA:
                    frames_abierta[idx] += 1
                    if frames_abierta[idx] >= MIN_FRAMES:
                        abierta[idx] = True
                        conteo[idx] += 1

                        # aquí se identifica un auto en este ROI
                        evento_roi[idx] = True
                else:
                    frames_abierta[idx] = 0
            else:
                # volver a estado cerrada cuando regresa cerca de baseline
                if abs(mean_val - baseline[idx]) < (DELTA_PLUMA / 2):
                    abierta[idx] = False
                    frames_abierta[idx] = 0

        # dibujar ROIs
        cv2.rectangle(frame_small, (0, pluma1_y1), (ancho, pluma1_y2), (0, 255, 0), 2)
        cv2.rectangle(frame_small, (0, pluma2_y1), (ancho, pluma2_y2), (255, 0, 0), 2)

        # texto con conteo TOTAL
        total_autos = conteo[0] + conteo[1]

        cv2.putText(
            frame_small,
            f"Autos totales: {total_autos}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # mostrar más pequeño el video
        display = cv2.resize(frame_small, (0, 0), fx=0.4, fy=0.4)
        cv2.imshow("Conteo por plumas (total)", display)

        # -------- MOSTRAR ROIs CON FILTRO DE MPI EN EL MOMENTO DEL AUTO --------

        # ROI pluma 1
        roi1 = gray[pluma1_y1:pluma1_y2, :]
        if evento_roi[0]:
            # aplicar filtro de convolución 2D con kernel de enfoque
            roi1_filt = mpi.convolucion2D(roi1, sharpen_kernel)
            roi1_show = cv2.cvtColor(roi1_filt, cv2.COLOR_GRAY2BGR)
        else:
            roi1_show = cv2.cvtColor(roi1, cv2.COLOR_GRAY2BGR)

        # ROI pluma 2
        roi2 = gray[pluma2_y1:pluma2_y2, :]
        if evento_roi[1]:
            roi2_filt = mpi.convolucion2D(roi2, sharpen_kernel)
            roi2_show = cv2.cvtColor(roi2_filt, cv2.COLOR_GRAY2BGR)
        else:
            roi2_show = cv2.cvtColor(roi2, cv2.COLOR_GRAY2BGR)

        cv2.imshow("ROI Pluma 1", cv2.resize(roi1_show, (0, 0), fx=0.8, fy=0.8))
        cv2.imshow("ROI Pluma 2", cv2.resize(roi2_show, (0, 0), fx=0.8, fy=0.8))


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    total_final = conteo[0] + conteo[1]
    print(f"Autos totales detectados: {total_final}")


if __name__ == "__main__":
    main()

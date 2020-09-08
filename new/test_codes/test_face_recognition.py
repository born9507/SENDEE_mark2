def face_reco():
    ##rgb_for_face 불러오기
    with open("pkl/rgb_for_face.pkl", "rb") as file:
        rgb_for_face = pickle.load(file)
        file.close()
    ##face_locations 불러오기
    with open("pkl/face_locations.pkl", "rb") as file:
        face_location = pickle.load(file)
        file.close()
    with open("pkl/known_face_names.pkl", "rb") as file:
        known_face_names = pickle.load(file)
        file.close()
    with open("pkl/known_face_encodings.pkl", "rb") as file:
        known_face_encodings = pickle.load(file)
        file.close()
    
    ##불러온 파일 이용해서 인코딩 구한다
    face_encoding = face_recognition.face_encodings(rgb_for_face, face_location)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding[0])
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding[0])
    best_match_index = np.argmin(face_distances)

    if matches[best_match_index]:
        name = known_face_names[best_match_index]
    else:
        name = "unknown"
    
    return name
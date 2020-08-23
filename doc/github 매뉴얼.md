## Github 매뉴얼

기본 명령어 

```git status
git status  //현재 상태 확인
git branch  //브랜치 목록, 원격 브랜치까지 다 보고싶다면 git branch -a
git checkout (브랜치 이름) //해당 브랜치 선택
```

```
git add .  //해당 디렉토리 내의 모든 폴더, 파일을 staging area에 올린다(임시저장 같은 느낌)
git commit -m "커밋 메시지"   // staging area 에 존재하는 변경사항을 저장한다
```

```
git pull  //원격 저장소의 최신 커밋을 local 로 가져온다. (연결된 브랜치에서)
git push  //local 저장소의 최신 커밋을 원격으로 업로드한다. (연결된 브랜치로)
```

커밋 시점이 다르면 충돌이 일어날 수 있다. 이럴 때는 vs code 등의 코드 편집기를 이용해 적절히 merge 해주고, 이후에 다시 커밋하여 변경사항을 적용한다.

master 브랜치는 정상 작동이 확인된 최종본만 올리는 것이 좋다. 사소한 변경사항은 브랜치를 갈라서 작업 후, pull request를 넣는 것이 안전한 협업 방법!

- https://github.com/born9507/SENDEE 에서 Folk 를 한다.

- (내 아이디)/SENDEE 라는 이름의 repository 가 생성된다

- `git clone https://github.com/(내 아이디)/SENDEE.git` 를 이용해 내 local 로 가져온다

- 클론한 폴더에 들어가서  `git remote -v `를 이용해 현재 연결된 원격 저장소를 확인한다

- `git remote add upstream https://github.com/born9507/SENDEE.git `  를 이용해 born9507의 원격 저장소를 upstream 이라는 이름으로 등록한다

  - 이는 원래의 repository 인 born9507/SENDEE 에 변화가 생겼을때 내려받기 위함이다
  - 내려받는 것은 `git pull upstream master` 라는 명령어로 수행한다. (upstream repository의 master 브랜치 내용을 가져온다는 뜻)

- 브랜치를 만드는 것은 두가지 방법이 있다

  - local에서: `git branch (브랜치 이름)` 하여 만든다. 그러나 이러면 원격에 연결된 브랜치가 없어서, 원격 브랜치를 만들고 연결도 해주어야 한다
  - github에서: 깃허브에서는 아래 사진처럼 브랜치를 클릭 후 새로운 브랜치 이름을 적는 것만으로 새로운 브랜치를 생성할 수 있다. 이후에 local 에서 `git pull` 을 통해 원격의 변경사항을 반영하고, `git checkout (원격에서 만든 브랜치 이름)` 하면, 자동으로 원격 브랜치와 연결된 로컬 브랜치가 생성된다.

  <img src="./img/newBranch.png" alt="``" style="zoom:60%;" />

**즉, 간단히 워크플로우를 써보자면**

1. git pull upstream master 를 통해 최신 변경사항을 적용
2. 로컬에서 작업
3. 커밋
4. git push 를 통해 내 원격저장소(origin) 으로 업로드
5. 깃허브에 들어가서 pull request 생성
6. 성공적으로 merge 되면 다시 1번부터 반복! 
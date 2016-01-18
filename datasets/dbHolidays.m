% Author: Relja Arandjelovic (relja@relja.info)

function db= dbHolidays(doRot)
    if nargin<1, doRot= false; end
    
    paths= localPaths();
    if doRot
        db.name= 'holidays_rot';
        db.dbPath= relja_expandUser([paths.dsetRootHolidays, 'jpg_rotated/']);
    else
        db.name= 'holidays';
        db.dbPath= relja_expandUser([paths.dsetRootHolidays, 'jpg/']);
    end
    db.qPath= db.dbPath;
    
    db.dbImageFns= relja_dir(db.dbPath);
    db.numImages= length(db.dbImageFns);
    assert(db.numImages==1491);
    db.queryIDs= find( cellfun(@(s) rem(str2num(s(1:(end-4))), 100)==0, db.dbImageFns) );
    db.numQueries= length(db.queryIDs);
    assert(db.numQueries==500);
    db.qImageFns= db.dbImageFns(db.queryIDs);
    
    db.posIDs= cell(db.numQueries, 1);
    db.ignoreIDs= cell( db.numQueries, 1 );
    tmp_queryIDs= [db.queryIDs; db.numImages+1];
    for iQuery= 1:db.numQueries
        db.ignoreIDs{iQuery}= db.queryIDs(iQuery);
        db.posIDs{iQuery}= (db.queryIDs(iQuery)+1):(tmp_queryIDs(iQuery+1)-1);
    end
    
    % Also potentially downscale to #pixels=1024x768, like in the original paper as well
    
    if doRot
        newDbPath= relja_expandUser([paths.dsetRootHolidays, 'jpg_rotated_1024x768/']);
    else
        newDbPath= relja_expandUser([paths.dsetRootHolidays, 'jpg_1024x768/']);
    end
    if ~exist(newDbPath, 'dir'), mkdir(newDbPath); end
    
    pixelNumThr= 1024*768;
    
    verbose= true;
    prog= tic;
    
    for iIm= 1:db.numImages
        if verbose
            relja_progress(iIm, db.numImages, 'check if resized exists, otherwise resize', prog);
        end
        fnNew= [newDbPath, db.dbImageFns{iIm}];
        if ~exist(fnNew, 'file')
            fnOrig= [db.dbPath, db.dbImageFns{iIm}];
            im= imread(fnOrig);
            pixelNum= size(im,1)*size(im,2);
            if pixelNum>pixelNumThr
                im= imresize( im, sqrt(pixelNumThr/pixelNum) );
                doWrite= true;
            else
                doWrite= false;
            end
            
            if doWrite
                imwrite(im, fnNew);
            else
                copyfile(fnOrig, fnNew);
            end
        end
    end
    db.dbPath= newDbPath;
    db.qPath= db.dbPath;
end
